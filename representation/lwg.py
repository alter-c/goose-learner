from .base_class import *
from planning.translate.pddl import Atom, NegatedAtom, Truth

from representation.deepwalk import *


class LWG_FEATURES(Enum):
    P = 0  # is predicate
    A = 1  # is action
    G = 2  # is positive goal (grounded)
    N = 3  # is negative goal (grounded)
    S = 4  # is activated (grounded)
    O = 5  # is object


ENC_FEAT_SIZE = len(LWG_FEATURES)
VAR_FEAT_SIZE = 4

LWG_EDGE_LABELS = OrderedDict(
    {
        "neutral": 0,   # predicate to object / action to args
        "ground": 1,    # grounded state and goal
        "pre_pos": 2,   # positive precondition
        "pre_neg": 3,   # negative precondition
        "eff_pos": 4,   # add effect
        "eff_neg": 5,   # del effect
    }
)


class LiftedWalkGraph(Representation, ABC):
    name = "lwg"
    n_node_features = ENC_FEAT_SIZE + VAR_FEAT_SIZE # semantics + variable features
    n_edge_labels = len(LWG_EDGE_LABELS)
    directed = False
    lifted = True

    def __init__(self, domain_pddl: str, problem_pddl: str):
        super().__init__(domain_pddl, problem_pddl)
        return

    def _construct_if(self) -> None:
        """Precompute a seeded randomly generated injective index function"""
        self._if = []
        image = set()  # check injectiveness

        max_idx = 60 # TODO read max range from problem and lazily compute
        for idx in range(max_idx):
            torch.manual_seed(idx)
            rep = 2 * torch.rand(VAR_FEAT_SIZE) - 1  # U[-1,1]
            rep /= torch.linalg.norm(rep)
            self._if.append(rep)
            key = tuple(rep.tolist())
            assert key not in image
            image.add(key)
        return
    
    def _feature(self, node_type: LWG_FEATURES) -> Tensor:
        ret = torch.zeros(self.n_node_features)
        ret[node_type.value] = 1
        return ret

    def _if_feature(self, idx: int) -> Tensor:
        ret = torch.zeros(self.n_node_features)
        ret[-VAR_FEAT_SIZE:] = self._if[idx]
        return ret

    def _compute_graph_representation(self) -> None:
        """构建图表示"""

        self._construct_if()

        G = self._create_graph()

        ### predicates
        largest_predicate = 0
        for pred in self.problem.predicates:
            largest_predicate = max(largest_predicate, len(pred.arguments))
            G.add_node(pred.name, x=self._feature(LWG_FEATURES.P))  # add predicate node

        ### actions
        largest_action_schema = 0
        for action in self.problem.actions:
            G.add_node(action.name, x=self._feature(LWG_FEATURES.A))
            action_args = {}

            largest_action_schema = max(largest_action_schema, len(action.parameters))
            for i, arg in enumerate(action.parameters):
                arg_node = (action.name, f"action-var-{i}")  # action param node
                G.add_node(arg_node, x=self._if_feature(idx=i))
                G.add_edge(
                    u_of_edge=action.name, v_of_edge=arg_node, edge_label=LWG_EDGE_LABELS["neutral"]
                )
                # store action param name and node mapping, for later predicate connection
                action_args[arg.name] = arg_node 

            def deal_with_action_prec_or_eff(predicates, edge_label):
                """deal with node and edge according to edge_label"""
                for z, predicate in enumerate(predicates):
                    pred = predicate.predicate
                    aux_node = (pred, f"{edge_label}-aux-{z}")  # node.name = pred.name + edge_label + index
                    G.add_node(aux_node, x=self._zero_node())   # aux node for duplicate preds (like grounded predicates)

                    assert pred in G.nodes()
                    G.add_edge(
                        u_of_edge=pred, v_of_edge=aux_node, edge_label=LWG_EDGE_LABELS[edge_label]
                    )

                    if len(predicate.args) > 0:
                        for j, arg in enumerate(predicate.args):
                            prec_arg_node = (arg, f"{edge_label}-aux-{z}-var-{j}") # node.name = var.name + edge_label + pred.index + var.index
                            G.add_node(prec_arg_node, x=self._if_feature(idx=j))    # aux var 
                            G.add_edge(
                                u_of_edge=aux_node,
                                v_of_edge=prec_arg_node,
                                edge_label=LWG_EDGE_LABELS[edge_label],
                            )

                            if arg in action_args:
                                action_arg_node = action_args[arg]
                                G.add_edge(
                                    u_of_edge=prec_arg_node,
                                    v_of_edge=action_arg_node,
                                    edge_label=LWG_EDGE_LABELS[edge_label],
                                )
                    else:  # unitary predicate so connect directly to action
                        G.add_edge(
                            u_of_edge=aux_node,
                            v_of_edge=action.name,
                            edge_label=LWG_EDGE_LABELS[edge_label],
                        )
                return

            # for precondition, only consider positive and negative predicates, not logical predicates like "or"
            # for effect, check: normal pred's condition is Truth. Thus no conditional effects.
            pos_pres = [p for p in action.precondition.parts if type(p) == Atom]
            neg_pres = [p for p in action.precondition.parts if type(p) == NegatedAtom]
            pos_effs = [p.literal for p in action.effects if type(p.literal) == Atom]
            neg_effs = [p.literal for p in action.effects if type(p.literal) == NegatedAtom]
            for p in action.effects:
                assert type(p.condition) == Truth  # no conditional effects

            # deal with precondition(pos/neg)和effect(add/del)
            deal_with_action_prec_or_eff(pos_pres, "pre_pos")
            deal_with_action_prec_or_eff(neg_pres, "pre_neg")
            deal_with_action_prec_or_eff(pos_effs, "eff_pos")
            deal_with_action_prec_or_eff(neg_effs, "eff_neg")
        ### end actions

        assert largest_predicate > 0
        assert largest_action_schema > 0

        # for same domain, domain nodes are fixed, save or load existed base deepwalk model
        MODEL_NAME = self.problem.domain_name
        domain_model = load_deepwalk_model(G, MODEL_NAME)
        # fixed basic feature and rearrange
        # self.domain_X = domain_model.wv.vectors[[domain_model.wv.key_to_index[str(node)] for node in G.nodes]]
        # assert self.domain_X.shape[1] == DEEPWALK_FEATURE_SIZE
        
        ### objects 
        object_nodes = [] # for incremental sampling and updating deepwalk model
        for i, obj in enumerate(self.problem.objects):
            G.add_node(obj.name, x=self._feature(LWG_FEATURES.O))  # add object node
            object_nodes.append(obj.name)
        
        ### fully connected between objects and predicates 
        for pred in self.problem.predicates:
            for obj in self.problem.objects:
                G.add_edge(
                    u_of_edge=pred.name, v_of_edge=obj.name, edge_label=LWG_EDGE_LABELS["neutral"]
                )

        ### goal (state gets dealt with in state_to_tensor) 
        goal_nodes = [] # same for incremental update
        if len(self.problem.goal.parts) == 0:
            goals = [self.problem.goal]
        else:
            goals = self.problem.goal.parts
        for fact in goals:
            assert type(fact) in {Atom, NegatedAtom}

            # may have negative goals
            is_negated = type(fact) == NegatedAtom

            pred = fact.predicate # predicate name
            args = fact.args      # grounded objects list
            goal_node = (pred, args)

            if is_negated: # according to the type of goal, add to pos/neg goal set
                x = self._feature(LWG_FEATURES.N)
                self._neg_goal_nodes.add(goal_node)
            else:
                x = self._feature(LWG_FEATURES.G)
                self._pos_goal_nodes.add(goal_node)
            G.add_node(goal_node, x=x)  # add grounded goal predicate node

            goal_nodes.append(goal_node)

            for i, arg in enumerate(args):
                goal_var_node = (goal_node, i)
                G.add_node(goal_var_node, x=self._if_feature(idx=i)) # add single goal variable node

                goal_nodes.append(goal_var_node)

                # connect variable to object
                G.add_edge(
                    u_of_edge=goal_node,
                    v_of_edge=goal_var_node,
                    edge_label=LWG_EDGE_LABELS["ground"],
                )

                # connect variable to object
                assert arg in G.nodes()
                G.add_edge(
                    u_of_edge=goal_var_node, v_of_edge=arg, edge_label=LWG_EDGE_LABELS["ground"]
                )

            # connect grounded fact to predicate
            assert pred in G.nodes()
            G.add_edge(u_of_edge=goal_node, v_of_edge=pred, edge_label=LWG_EDGE_LABELS["ground"])
        ### end goal

        # map node name to index
        self._node_to_i = {}
        for i, node in enumerate(G.nodes):
            self._node_to_i[node] = i
        self.G = G

        # for same problem, domain ,object and goal nodes are fixed, update base model for specific problem
        new_nodes = object_nodes + goal_nodes 
        self.model = incremental_deepwalk(G, new_nodes, domain_model) 

        return

    def str_to_state(self, s) -> List[Tuple[str, List[str]]]:
        """Used in dataset construction to convert string representation of facts into a (pred, [args]) representation"""
        state = []
        for fact in s:
            fact = fact.replace(")", "").replace("(", "")
            toks = fact.split()
            if toks[0] == "=":
                continue
            if len(toks) > 1:
                state.append((toks[0], toks[1:]))
            else:
                state.append((toks[0], ()))
        return state

    def state_to_tensor(self, state: List[Tuple[str, List[str]]]) -> TGraph:
        """
        States are represented as a list of (pred, [args])
        semantics features: state -> tensor
        deepwalk features: state -> graph -> deepwalk features -> tensor

        # why deal with state separately?
        1. only state changes while the rest of the graph remains static
        2. reduce the computation load of graph construction
        """
        ### computing semantics features
        x = self.x.clone()
        edge_indices = self.edge_indices.copy()
        i = len(x)

        to_add = sum(len(fact[1]) + 1 for fact in state)
        x = torch.nn.functional.pad(x, (0, 0, 0, to_add), "constant", 0)
        append_edge_index = []

        state_nodes = [] # for incremental sampling and updating deepwalk model

        for fact in state:
            pred = fact[0]
            args = fact[1]

            if len(pred) == 0:
                continue # somehow get ("", []) facts, empty pred will cause bug when add edge

            node = (pred, tuple(args))

            # activated proposition overlaps with a goal Atom or NegatedAtom
            if node in self._node_to_i:
                x[self._node_to_i[node]][LWG_FEATURES.S.value] = 1
                continue

            # activated proposition does not overlap with a goal
            true_node_i = i
            x[i][LWG_FEATURES.S.value] = 1
            i += 1

            state_nodes.append(node) 

            # connect fact to predicate
            append_edge_index.append((true_node_i, self._node_to_i[pred]))
            append_edge_index.append((self._node_to_i[pred], true_node_i))

            # connect to predicates and objects
            for k, arg in enumerate(args):
                arg_node = (node, f"true-var-{k}")

                true_var_node_i = i
                x[i][-VAR_FEAT_SIZE:] = self._if[k]
                i += 1
                
                state_nodes.append(arg_node) 

                # connect variable to predicate
                append_edge_index.append((true_node_i, true_var_node_i))
                append_edge_index.append((true_var_node_i, true_node_i))

                # connect variable to object
                append_edge_index.append((true_var_node_i, self._node_to_i[arg]))
                append_edge_index.append((self._node_to_i[arg], true_var_node_i))

        # cut off unused part of x 
        x = x[:i]
        # incremental update edge indices
        edge_indices[LWG_EDGE_LABELS["ground"]] = torch.hstack(
            (edge_indices[LWG_EDGE_LABELS["ground"]], torch.tensor(append_edge_index).T)
        ).long()


        ### computing deepwalk features
        state_graph = self.state_to_cgraph(state)
        model = incremental_deepwalk(state_graph, state_nodes, self.model)

        # rearrange the deepwalk features according to the original node order
        X = model.wv.vectors[[model.wv.key_to_index[str(node)] for node in state_graph.nodes]]
        X = torch.tensor(X) # convert to torch tensor

        feature = torch.cat((x, X), dim=1) # concatenate semantics and deepwalk features

        return feature, edge_indices
        
    def state_to_cgraph(self, state: List[Tuple[str, List[str]]]) -> CGraph:
        """
        transform state to graph
        this graph is just used for generating deepwalk features, no semantics features
        """
        # state graph for generating deepwalk features
        state_graph = self.G.copy()

        for fact in state:
            pred = fact[0]
            args = fact[1]

            if len(pred) == 0:
                continue # somehow get ("", []) facts, empty pred will cause bug when add edge

            node = (pred, tuple(args))

            # activated proposition overlaps with a goal Atom or NegatedAtom
            if node in self._node_to_i:
                continue

            # activated proposition does not overlap with a goal
            state_graph.add_node(node)

            # connect fact to predicate
            state_graph.add_edge(u_of_edge=node, v_of_edge=pred, edge_label=LWG_EDGE_LABELS["ground"])

            # connect to predicates and objects
            for k, arg in enumerate(args):
                arg_node = (node, f"true-var-{k}")
                state_graph.add_node(arg_node)

                # connect variable to predicate
                state_graph.add_edge(
                    u_of_edge=node, v_of_edge=arg_node, edge_label=LWG_EDGE_LABELS["ground"]
                )
                # connect variable to object
                state_graph.add_edge(
                    u_of_edge=arg_node, v_of_edge=arg, edge_label=LWG_EDGE_LABELS["ground"]
                )

        return state_graph
    