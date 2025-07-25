from collections import defaultdict
from functools import lru_cache
from random import choice

from codex.data.connections import Connections
from codex.data.neurotransmitters import NEURO_TRANSMITTER_NAMES

from codex.data.search_index import SearchIndex
from codex.data.structured_search_filters import (
    make_structured_terms_predicate,
    apply_chaining_rule,
    parse_search_query,
)
from codex.configuration import MIN_NBLAST_SCORE_SIMILARITY
from codex.utils.formatting import (
    display,
    percentage,
)

from codex.utils.stats import jaccard_weighted, jaccard_binary

from codex import logger

NEURON_SEARCH_LABEL_ATTRIBUTES = [
    "root_id",
    "name",
    "group",
    "nt_type",
    "super_class",
    "class",
    "sub_class",
    "cell_type",
    "flow",
    "hemilineage",
    "nerve",
    "side",
    "connectivity_tag",
]


class NeuronDB(object):
    def __init__(
        self,
        neuron_attributes,
        neuron_connection_rows,
        label_data,
        labels_file_timestamp,
        grouped_synapse_counts,
        grouped_connection_counts,
        grouped_reciprocal_connection_counts,
    ):
        self.neuron_data = neuron_attributes
        self.connections_ = Connections(neuron_connection_rows)
        self.label_data = label_data
        self.grouped_synapse_counts = grouped_synapse_counts
        self.grouped_connection_counts = grouped_connection_counts
        self.grouped_reciprocal_connection_counts = grouped_reciprocal_connection_counts
        self.meta_data = {"labels_file_timestamp": labels_file_timestamp}

        logger.debug("App initialization building search index..")

        def searchable_labels(ndata):
            labels = []
            for c in NEURON_SEARCH_LABEL_ATTRIBUTES:
                val = ndata[c]
                if val:
                    if isinstance(val, list):
                        labels += val
                    else:
                        labels.append(val)
            return labels

        self.search_index = SearchIndex(
            [
                (nd["label"], searchable_labels(nd), k)
                for k, nd in self.neuron_data.items()
            ]
        )

    def input_sets(self, min_syn_count=0):
        return self.input_output_partner_sets(min_syn_count)[0]

    def output_sets(self, min_syn_count=0):
        return self.input_output_partner_sets(min_syn_count)[1]

    @lru_cache
    def input_output_partner_sets(self, min_syn_count=0):
        ins, outs = self.input_output_partners_with_synapse_counts(
            min_syn_count=min_syn_count
        )
        ins = {k: set(v.keys()) for k, v in ins.items()}
        outs = {k: set(v.keys()) for k, v in outs.items()}
        return ins, outs

    @lru_cache
    def input_output_partners_with_synapse_counts(self, min_syn_count=0):
        ins, outs = self.connections_.input_output_partners_with_synapse_counts()
        if min_syn_count:

            def apply_syn_threshold(syn_counts):
                return {
                    rid_: cnt
                    for rid_, cnt in syn_counts.items()
                    if cnt >= min_syn_count
                }

            ins = {k: apply_syn_threshold(v) for k, v in ins.items()}
            outs = {k: apply_syn_threshold(v) for k, v in outs.items()}

        for rid in self.neuron_data.keys():
            if rid not in ins:
                ins[rid] = {}
            if rid not in outs:
                outs[rid] = {}
        return ins, outs

    @lru_cache
    def input_output_regions_with_synapse_counts(self):
        ins, outs = self.connections_.input_output_regions_with_synapse_counts()
        for rid in self.neuron_data.keys():
            if rid not in ins:
                ins[rid] = {}
            if rid not in outs:
                outs[rid] = {}
        return ins, outs

    @lru_cache
    def cell_connections(self, cell_id):
        return list(self.connections_.rows_for_cell(cell_id))

    def connections(
        self, ids, induced=False, min_syn_count=0, nt_type=None, regions=None
    ):
        if nt_type and nt_type not in NEURO_TRANSMITTER_NAMES:
            raise ValueError(
                f"Unknown NT type: {nt_type}, must be one of {NEURO_TRANSMITTER_NAMES}"
            )
        if induced:
            return list(
                self.connections_.rows_between_sets(
                    ids,
                    ids,
                    min_syn_count=min_syn_count,
                    nt_type=nt_type,
                    regions=regions,
                )
            )
        else:
            return list(
                self.connections_.rows_for_set(
                    ids, min_syn_count=min_syn_count, nt_type=nt_type, regions=regions
                )
            )

    @lru_cache
    def connections_up_down(self, cell_id, by_neuropil=False):
        try:
            cell_id = int(cell_id)
        except ValueError:
            raise ValueError(f"'{cell_id}' is not a valid cell ID")
        table = self.cell_connections(cell_id)
        if by_neuropil:
            downstream = defaultdict(list)
            upstream = defaultdict(list)
        else:
            downstream = []
            upstream = []
        for r in table or []:
            if r[0] == cell_id:
                if by_neuropil:
                    downstream[r[2]].append(r[1])
                else:
                    downstream.append(r[1])
            else:
                assert r[1] == cell_id
                if by_neuropil:
                    upstream[r[2]].append(r[0])
                else:
                    upstream.append(r[0])
        return downstream, upstream

    def random_cell_id(self):
        return choice(list(self.neuron_data.keys()))

    def num_cells(self):
        return len(self.neuron_data)

    def num_synapses(self):
        return self.connections_.num_synapses()

    def num_connections(self):
        return self.connections_.num_connections()

    @lru_cache
    def num_labels(self):
        return sum([len(nd["label"]) for nd in self.neuron_data.values()])

    @lru_cache
    def num_typed_or_identified_cells(self):
        return len(
            [
                nd
                for nd in self.neuron_data.values()
                if any([nd[attr] for attr in ["label", "cell_type"]])
            ]
        )

    @lru_cache
    def unique_values(self, attr_name):
        vals = set()
        for nd in self.neuron_data.values():
            if nd[attr_name]:
                if isinstance(nd[attr_name], list):
                    vals |= set(nd[attr_name])
                else:
                    vals.add(nd[attr_name])
        return sorted(vals)

    @lru_cache
    def categories(self, top_values, for_attr_name=None):
        value_counts_dict = defaultdict(lambda: defaultdict(int))
        assigned_to_num_cells_dict = defaultdict(int)
        category_attr_names = {
            "Neurotransmitter Type": "nt_type",
            "Flow": "flow",
            "Super Class": "super_class",
            "Class": "class",
            "Sub Class": "sub_class",
            "Cell Type": "cell_type",
            "Hemilineage": "hemilineage",
            "Nerve": "nerve",
            "Cell Body Side": "side",
            "Community Identification Label": "label",
            "Connectivity Tag": "connectivity_tag",
            "Max In/Out Neuropil": "group",
        }
        for nd in self.neuron_data.values():
            for cat_attr in category_attr_names.values():
                if "." in cat_attr:
                    prts = cat_attr.split(".")
                    v, v_prefix = prts[0], prts[1]
                else:
                    v, v_prefix = cat_attr, None

                if for_attr_name and v != for_attr_name:
                    continue

                val = nd[v]
                if not val:
                    continue

                if isinstance(val, list):
                    assigned = False
                    for c in val:
                        if v_prefix:
                            if c.startswith(f"{v_prefix}:"):
                                c = c[len(f"{v_prefix}:") :]
                            else:
                                continue
                        assigned = True
                        value_counts_dict[v][c] += 1
                else:
                    assigned = True
                    value_counts_dict[v][val] += 1

                if assigned:
                    assigned_to_num_cells_dict[v] += 1

        def _caption(name, assigned_to_count, values_count):
            caption = (
                f"<b>{name}</b><small style='color: teal'>"
                f"<br>- Assigned to {display(assigned_to_count)} cells / {percentage(assigned_to_count, self.num_cells())}"
                f"<br>- {display(values_count)} unique values"
            )
            if values_count > top_values:
                caption += f". Showing top {top_values}</small> <b>&#8594;</b>"
            else:
                caption += "</small>"
            return caption

        def _sorted_counts(d):
            lst_all = sorted([(k, v) for k, v in d.items() if k], key=lambda p: -p[1])
            return lst_all[:top_values]

        return [
            {
                "caption": _caption(
                    ck, assigned_to_num_cells_dict[cv], len(value_counts_dict[cv])
                ),
                "key": cv,
                "counts": _sorted_counts(value_counts_dict[cv]),
            }
            for ck, cv in category_attr_names.items()
            if value_counts_dict[cv]
        ]

    # Returns value ranges for all attributes with not too many different values. Used for advanced search dropdowns.
    @lru_cache
    def dynamic_ranges(self, range_cardinality_cap=40):
        res = {}
        for dct in self.categories(top_values=range_cardinality_cap):
            if len(dct["counts"]) < range_cardinality_cap:
                res[f"data_{dct['key']}_range"] = [p[0] for p in dct["counts"]]
        return res

    def is_in_dataset(self, root_id):
        root_id = int(root_id)
        return root_id in self.neuron_data

    def get_neuron_data(self, root_id):
        root_id = int(root_id)
        nd = self.neuron_data.get(root_id)
        if not nd:
            logger.debug(
                f"No data exists for {root_id} in {len(self.neuron_data)} records"
            )
            nd = {}
        return nd

    @lru_cache
    def get_similar_shape_cells(
        self,
        root_id,
        include_self,
        min_score=MIN_NBLAST_SCORE_SIMILARITY,
        top_k=99999,
    ):
        scores = (
            [
                (rid, score)
                for rid, score in self.get_neuron_data(root_id)[
                    "similar_cell_scores"
                ].items()
                if score >= min_score
            ]
            if self.get_neuron_data(root_id).get("similar_cell_scores")
            else []
        )
        if include_self:
            scores.append((root_id, 10))
        scores = sorted(scores, key=lambda p: -p[1])[:top_k]
        return {p[0]: p[1] for p in scores}

    @lru_cache
    def get_similar_connectivity_cells(
        self,
        root_id,
        include_upstream=True,
        include_downstream=True,
        weighted=False,
        with_same_attributes=None,
        include_score_threshold=0.2,
        min_score_threshold=0.1,
        min_score_limit=20,
    ):
        if not self.is_in_dataset(root_id):
            raise ValueError(
                f"{root_id} is not a valid cell ID or not included in this data snapshot."
            )
        if weighted:
            ins, outs = self.input_output_partners_with_synapse_counts()
            jaccard_score = jaccard_weighted
            upstream_filter_attr_name = "input_synapses"
            downstream_filter_attr_name = "output_synapses"
        else:
            ins, outs = self.input_output_partner_sets()
            jaccard_score = jaccard_binary
            upstream_filter_attr_name = "input_cells"
            downstream_filter_attr_name = "output_cells"

        def calc_range_for_threshold(attr_name):
            val = self.get_neuron_data(root_id)[attr_name]
            return val * min_score_threshold, val / min_score_threshold

        upstream_filter_lb, upstream_filter_ub = calc_range_for_threshold(
            upstream_filter_attr_name
        )
        downstream_filter_lb, downstream_filter_ub = calc_range_for_threshold(
            downstream_filter_attr_name
        )

        if with_same_attributes:
            match_attributes = {
                attr: self.neuron_data[root_id][attr]
                for attr in with_same_attributes.split(",")
            }
        else:
            match_attributes = None

        def filter_out(nd):
            # optimization filters
            if include_upstream and not (
                upstream_filter_lb
                <= nd[upstream_filter_attr_name]
                <= upstream_filter_ub
            ):
                return True
            if include_downstream and not (
                downstream_filter_lb
                <= nd[downstream_filter_attr_name]
                <= downstream_filter_ub
            ):
                return True
            if match_attributes and not all(
                [nd[attr] == val for attr, val in match_attributes.items()]
            ):
                return True
            return False

        def calc_similarity_score(r, nd):
            if filter_out(nd):
                return 0
            combined_score, num_scores = 0, 0
            if include_upstream:
                combined_score += jaccard_score(ins[root_id], ins[r])
                num_scores += 1
            if include_downstream:
                combined_score += jaccard_score(outs[root_id], outs[r])
                num_scores += 1
            return combined_score / num_scores

        scores = []
        for rid, ndata in self.neuron_data.items():
            score = calc_similarity_score(rid, ndata)
            if score >= min_score_threshold:
                scores.append((rid, score))
        scores = sorted(scores, key=lambda p: -p[1])
        res = {}
        for p in scores:
            if p[1] >= include_score_threshold or len(res) < min_score_limit:
                res[p[0]] = p[1]
            else:
                break
        return res

    def get_all_cell_types(self, root_id):
        nd = self.get_neuron_data(root_id)
        return nd["cell_type"]

    def get_label_data(self, root_id):
        root_id = int(root_id)
        return self.label_data.get(root_id)

    def label_data_for_ids(self, ids, user_filter=None, lab_filter=None):
        if user_filter:
            user_filter = user_filter.lower()
        if lab_filter:
            lab_filter = lab_filter.lower()

        def filtered(label_list):
            if user_filter:
                label_list = [
                    ld
                    for ld in label_list
                    if ld["user_name"] and user_filter in ld["user_name"].lower()
                ]
            if lab_filter:
                label_list = [
                    ld
                    for ld in label_list
                    if ld["user_affiliation"]
                    and lab_filter in ld["user_affiliation"].lower()
                ]
            return label_list

        res = {}
        for r in ids:
            flist = filtered(self.label_data[r])
            if flist:
                res[r] = flist
        return res

    def get_links(self, root_id):
        nd = self.get_neuron_data(root_id)
        links = []
        for mrk in nd["marker"]:
            if mrk.startswith("link:"):
                links.append(mrk[len("link:") :])
        return links

    def cell_ids_with_label_data(self):
        return list(self.label_data.keys())

    def labels_ingestion_timestamp(self):
        return self.meta_data["labels_file_timestamp"]

    @lru_cache
    def search(self, search_query, case_sensitive=False, word_match=False):
        if not search_query:
            return sorted(
                self.neuron_data.keys(),
                key=lambda rid: self.neuron_data[rid]["input_cells"]
                + self.neuron_data[rid]["output_cells"],
                reverse=True,
            )

        # The basic search query term can be either "free form" or "structured".
        # - Free form is when user types in a keyword, or a sentence, and the goal is to find all items that match
        #   (w.r.t any of their searchable attributes).
        # - Structured is something like 'label == olfactory' or 'nt_type != GABA' etc.
        #
        # These basic terms can be chained with one of the chaining rules: '&&' (and), '||' (or).
        # For example, 'label == olfactory && nt_type != GABA' should match items where label is olfactory and NT type
        # is other than GABA. Similarly, 'JON || nt_type == GABA' should find anything that matches JON (free form) or
        # has NT equal to GABA (structured).
        #
        # For chained search queries, we execute all 'free form' parts separately, and we combine one predicate for the
        # 'structured' parts to be evaluated once on every item. This is an optimization, because free form queries are
        # index lookups (and quick), while structured queries are evaluated in a linear scan. Then we combine the
        # collected results with the chaining rule (intersection for '&&' / union for '||').

        chaining_rule, free_form_terms, structured_terms = parse_search_query(
            search_query
        )
        term_search_results = []
        for term in free_form_terms:
            matching_results = self.search_index.search(
                term=term, case_sensitive=case_sensitive, word_match=word_match
            )
            term_search_results.append(matching_results)

        if structured_terms:
            predicate = make_structured_terms_predicate(
                chaining_rule=chaining_rule,
                structured_terms=structured_terms,
                input_sets_getter=self.input_sets,
                output_sets_getter=self.output_sets,
                connections_loader=self.connections_up_down,
                similar_cells_loader=self.get_similar_shape_cells,
                similar_connectivity_loader=self.get_similar_connectivity_cells,
                case_sensitive=case_sensitive,
            )
            term_search_results.append(
                [k for k, v in self.neuron_data.items() if predicate(v)]
            )

        return apply_chaining_rule(
            chaining_rule=chaining_rule, term_search_results=term_search_results
        )

    def closest_token(self, query, case_sensitive, limited_ids_set=None):
        query = query.strip()
        if not query or query.isnumeric():  # do not suggest number/id close matches
            return None, None
        chaining_rule, free_form_terms, structured_terms = parse_search_query(query)
        if chaining_rule or structured_terms:  # do not suggest for structured queries
            return None, None
        return self.search_index.closest_token(
            term=query, case_sensitive=case_sensitive, limited_ids_set=limited_ids_set
        )

    def multi_val_attrs(self, ids):
        # Given list of cell ids, returns the attrs for which the set of values in these cells is >1
        # This is used for deciding when to allow "include / exclude" filters.

        attr_vals = defaultdict(set)
        candidate_attr_names = {
            "super_class",
            "class",
            "sub_class",
            "hemilineage",
            "flow",
            "side",
            "nt_type",
            "nerve",
        }
        multi_val_attr_names = set()

        for cell_id in ids:
            nd = self.neuron_data[cell_id]
            for attr_name in candidate_attr_names:
                attr_vals[attr_name].add(nd[attr_name])
                if len(attr_vals[attr_name]) > 1:
                    multi_val_attr_names.add(attr_name)
            candidate_attr_names -= multi_val_attr_names
            if not candidate_attr_names:
                break

        return multi_val_attr_names

    def non_uniform_values(self, list_attr_key, page_ids, all_ids):
        # Returns the attr_vals for cells in page_ids such that some cells from all_ids have them while others don't.
        # This is used for deciding when to allow "include / exclude" filters.
        page_attr_vals = set()
        for cell_id in page_ids:
            for val in self.neuron_data[cell_id][list_attr_key]:
                page_attr_vals.add(val)
        non_uniform_set = set()
        for i, cell_id in enumerate(all_ids):
            val_set = set(self.neuron_data[cell_id][list_attr_key])
            non_uniform_set |= page_attr_vals - val_set
            if len(non_uniform_set) == len(page_attr_vals):
                break
        return non_uniform_set
