from .base_class import *
from planning.translate.pddl import Atom, NegatedAtom, Truth

# TODO 需要将目标与状态统一, 都视为状态
class CLG_FEATURES(Enum):
    P = 0  # is predicate
    A = 1  # is action
    G = 2  # is positive goal (grounded)
    N = 3  # is negative goal (grounded)
    S = 4  # is activated (grounded)
    O = 5  # is object


ENC_FEAT_SIZE = len(CLG_FEATURES)
VAR_FEAT_SIZE = 4

CLG_EDGE_LABELS = OrderedDict(
    {
        "neutral": 0,
        "ground": 1,
        "pre_pos": 2,
        "pre_neg": 3,
        "eff_pos": 4,
        "eff_neg": 5,
    }
)


class ContrastiveLearningGraph(Representation, ABC):
    name = "clg"
    n_node_features = ENC_FEAT_SIZE + VAR_FEAT_SIZE
    n_edge_labels = len(CLG_EDGE_LABELS)
    directed = False
    lifted = True

    def __init__(self, domain_pddl: str, problem_pddl: str):
        super().__init__(domain_pddl, problem_pddl)

    def _construct_if(self) -> None:
        """Precompute a seeded randomly generated injective index function"""
        self._if = []
        image = set()  # check injectiveness

        # TODO read max range from problem and lazily compute
        for idx in range(60):
            torch.manual_seed(idx)
            rep = 2 * torch.rand(VAR_FEAT_SIZE) - 1  # U[-1,1]
            rep /= torch.linalg.norm(rep)
            self._if.append(rep)
            key = tuple(rep.tolist())
            assert key not in image
            image.add(key)
        return

    def _feature(self, node_type: CLG_FEATURES) -> Tensor:
        ret = torch.zeros(self.n_node_features)
        ret[node_type.value] = 1
        return ret

    def _if_feature(self, idx: int) -> Tensor:
        ret = torch.zeros(self.n_node_features)
        ret[-VAR_FEAT_SIZE:] = self._if[idx]
        return ret

    def _compute_graph_representation(self) -> None:
        """TODO: reference definition of this graph representation"""

        self._construct_if()

        G = self._create_graph()

        # objects
        for i, obj in enumerate(self.problem.objects):
            G.add_node(obj.name, x=self._feature(CLG_FEATURES.O))  # add object node

        # predicates
        largest_predicate = 0
        for pred in self.problem.predicates:
            largest_predicate = max(largest_predicate, len(pred.arguments))
            G.add_node(pred.name, x=self._feature(CLG_FEATURES.P))  # add predicate node

        # fully connected between objects and predicates
        for pred in self.problem.predicates:
            for obj in self.problem.objects:
                G.add_edge(
                    u_of_edge=pred.name, v_of_edge=obj.name, edge_label=CLG_EDGE_LABELS["neutral"]
                )

        # 对状态编码，因此初始化时不对目标编码
        """
        # goal (state gets dealt with in state_to_tensor)
        if len(self.problem.goal.parts) == 0:
            goals = [self.problem.goal]
        else:
            goals = self.problem.goal.parts
        for fact in goals:
            assert type(fact) in {Atom, NegatedAtom}

            # may have negative goals
            is_negated = type(fact) == NegatedAtom

            pred = fact.predicate
            args = fact.args
            goal_node = (pred, args)

            if is_negated:
                x = self._feature(CLG_FEATURES.N)
                self._neg_goal_nodes.add(goal_node)
            else:
                x = self._feature(CLG_FEATURES.G)
                self._pos_goal_nodes.add(goal_node)
            G.add_node(goal_node, x=x)  # add grounded predicate node

            for i, arg in enumerate(args):
                goal_var_node = (goal_node, i)
                G.add_node(goal_var_node, x=self._if_feature(idx=i))

                # connect variable to predicate
                G.add_edge(
                    u_of_edge=goal_node,
                    v_of_edge=goal_var_node,
                    edge_label=CLG_EDGE_LABELS["ground"],
                )

                # connect variable to object
                assert arg in G.nodes()
                G.add_edge(
                    u_of_edge=goal_var_node, v_of_edge=arg, edge_label=CLG_EDGE_LABELS["ground"]
                )

            # connect grounded fact to predicate
            assert pred in G.nodes()
            G.add_edge(u_of_edge=goal_node, v_of_edge=pred, edge_label=CLG_EDGE_LABELS["ground"])
        # end goal
        """

        # actions
        largest_action_schema = 0
        for action in self.problem.actions:
            G.add_node(action.name, x=self._feature(CLG_FEATURES.A))
            action_args = {}

            largest_action_schema = max(largest_action_schema, len(action.parameters))
            for i, arg in enumerate(action.parameters):
                arg_node = (action.name, f"action-var-{i}")  # action var
                G.add_node(arg_node, x=self._if_feature(idx=i))
                action_args[arg.name] = arg_node
                G.add_edge(
                    u_of_edge=action.name, v_of_edge=arg_node, edge_label=CLG_EDGE_LABELS["neutral"]
                )

            def deal_with_action_prec_or_eff(predicates, edge_label):
                for z, predicate in enumerate(predicates):
                    pred = predicate.predicate
                    aux_node = (pred, f"{edge_label}-aux-{z}")  # aux node for duplicate preds
                    G.add_node(aux_node, x=self._zero_node())

                    assert pred in G.nodes()
                    G.add_edge(
                        u_of_edge=pred, v_of_edge=aux_node, edge_label=CLG_EDGE_LABELS[edge_label]
                    )

                    if len(predicate.args) > 0:
                        for j, arg in enumerate(predicate.args):
                            prec_arg_node = (arg, f"{edge_label}-aux-{z}-var-{j}")  # aux var
                            G.add_node(prec_arg_node, x=self._if_feature(idx=j))
                            G.add_edge(
                                u_of_edge=aux_node,
                                v_of_edge=prec_arg_node,
                                edge_label=CLG_EDGE_LABELS[edge_label],
                            )

                            if arg in action_args:
                                action_arg_node = action_args[arg]
                                G.add_edge(
                                    u_of_edge=prec_arg_node,
                                    v_of_edge=action_arg_node,
                                    edge_label=CLG_EDGE_LABELS[edge_label],
                                )
                    else:  # unitary predicate so connect directly to action
                        G.add_edge(
                            u_of_edge=aux_node,
                            v_of_edge=action.name,
                            edge_label=CLG_EDGE_LABELS[edge_label],
                        )
                return

            pos_pres = [p for p in action.precondition.parts if type(p) == Atom]
            neg_pres = [p for p in action.precondition.parts if type(p) == NegatedAtom]
            pos_effs = [p.literal for p in action.effects if type(p.literal) == Atom]
            neg_effs = [p.literal for p in action.effects if type(p.literal) == NegatedAtom]
            for p in action.effects:
                assert type(p.condition) == Truth  # no conditional effects
            deal_with_action_prec_or_eff(pos_pres, "pre_pos")
            deal_with_action_prec_or_eff(neg_pres, "pre_neg")
            deal_with_action_prec_or_eff(pos_effs, "eff_pos")
            deal_with_action_prec_or_eff(neg_effs, "eff_neg")
        # end actions

        assert largest_predicate > 0
        assert largest_action_schema > 0

        # map node name to index
        self._node_to_i = {}
        for i, node in enumerate(G.nodes):
            self._node_to_i[node] = i
        self.G = G

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

    def goal_to_tensor(self) -> TGraph:
        """单独计算goal的tensor"""
        G = self.G.copy()
        
        # goal (state gets dealt with in state_to_tensor)
        if len(self.problem.goal.parts) == 0:
            goals = [self.problem.goal]
        else:
            goals = self.problem.goal.parts
        for fact in goals:
            assert type(fact) in {Atom, NegatedAtom}

            # may have negative goals
            is_negated = type(fact) == NegatedAtom

            pred = fact.predicate
            args = fact.args
            goal_node = (pred, args)

            if is_negated:
                x = self._feature(CLG_FEATURES.N)
                self._neg_goal_nodes.add(goal_node)
            else:
                x = self._feature(CLG_FEATURES.G)
                self._pos_goal_nodes.add(goal_node)
            G.add_node(goal_node, x=x)  # add grounded predicate node

            for i, arg in enumerate(args):
                goal_var_node = (goal_node, i)
                G.add_node(goal_var_node, x=self._if_feature(idx=i))

                # connect variable to predicate
                G.add_edge(
                    u_of_edge=goal_node,
                    v_of_edge=goal_var_node,
                    edge_label=CLG_EDGE_LABELS["ground"],
                )

                # connect variable to object
                assert arg in G.nodes()
                G.add_edge(
                    u_of_edge=goal_var_node, v_of_edge=arg, edge_label=CLG_EDGE_LABELS["ground"]
                )

            # connect grounded fact to predicate
            assert pred in G.nodes()
            G.add_edge(u_of_edge=goal_node, v_of_edge=pred, edge_label=CLG_EDGE_LABELS["ground"])
        # end goal

        return self.convert_to_TG(G)

    def state_to_tensor(self, state: List[Tuple[str, List[str]]]) -> TGraph:
        """States are represented as a list of (pred, [args])"""
        # 状态更新到最后，会达到目标状态，因此目标状态也会编码
        # 另一方面, clg后续对比学习计算语义特征，而非直接计算启发值，因此不需要在特征里加入目标信息

        x = self.x.clone()
        edge_indices = self.edge_indices.copy()
        i = len(x)

        to_add = sum(len(fact[1]) + 1 for fact in state)
        x = torch.nn.functional.pad(x, (0, 0, 0, to_add), "constant", 0)
        append_edge_index = []

        for fact in state:
            pred = fact[0]
            args = fact[1]
            if len(pred) == 0:
                continue  # somehow we get ("", []) facts???

            node = (pred, tuple(args))

            # 单纯对状态编码，无需检查目标
            # if node in self._node_to_i:
            #     x[self._node_to_i[node]][CLG_FEATURES.S.value] = 1
            #     continue

            # activated proposition does not overlap with a goal
            true_node_i = i
            x[i][CLG_FEATURES.S.value] = 1
            i += 1

            # connect fact to predicate
            append_edge_index.append((true_node_i, self._node_to_i[pred]))
            append_edge_index.append((self._node_to_i[pred], true_node_i))

            # connect to predicates and objects
            for k, arg in enumerate(args):
                true_var_node_i = i
                x[i][-VAR_FEAT_SIZE:] = self._if[k]
                i += 1

                # connect variable to predicate
                append_edge_index.append((true_node_i, true_var_node_i))
                append_edge_index.append((true_var_node_i, true_node_i))

                # connect variable to object
                append_edge_index.append((true_var_node_i, self._node_to_i[arg]))
                append_edge_index.append((self._node_to_i[arg], true_var_node_i))

        edge_indices[CLG_EDGE_LABELS["ground"]] = torch.hstack(
            (edge_indices[CLG_EDGE_LABELS["ground"]], torch.tensor(append_edge_index).T)
        ).long()

        return x, edge_indices

    def state_to_cgraph(self, state: List[Tuple[str, List[str]]]) -> CGraph:
        """States are represented as a list of (pred, [args])"""
        c_graph = self.c_graph.copy()

        for fact in state:
            pred = fact[0]
            args = fact[1]

            node = (pred, tuple(args))

            # activated proposition overlaps with a goal Atom or NegatedAtom
            if node in self._pos_goal_nodes:
                c_graph.nodes[node]["colour"] = (
                    c_graph.nodes[node]["colour"] + ACTIVATED_POS_GOAL_COLOUR_SUFFIX
                )
                continue
            elif node in self._neg_goal_nodes:
                c_graph.nodes[node]["colour"] = (
                    c_graph.nodes[node]["colour"] + ACTIVATED_NEG_GOAL_COLOUR_SUFFIX
                )
                continue

            # else add node and corresponding edges to graph
            c_graph.add_node(node, colour=ACTIVATED_COLOUR)

            # connect fact to predicate
            c_graph.add_edge(u_of_edge=node, v_of_edge=pred, edge_label=CLG_EDGE_LABELS["ground"])
            c_graph.add_edge(v_of_edge=node, u_of_edge=pred, edge_label=CLG_EDGE_LABELS["ground"])

            # connect to predicates and objects
            for k, arg in enumerate(args):
                arg_node = (node, f"true-var-{k}")
                c_graph.add_node(arg_node, colour=str(k) + IF_COLOUR_SUFFIX)

                # connect variable to predicate
                c_graph.add_edge(
                    u_of_edge=node, v_of_edge=arg_node, edge_label=CLG_EDGE_LABELS["ground"]
                )
                c_graph.add_edge(
                    v_of_edge=node, u_of_edge=arg_node, edge_label=CLG_EDGE_LABELS["ground"]
                )

                # connect variable to object
                c_graph.add_edge(
                    u_of_edge=arg_node, v_of_edge=arg, edge_label=CLG_EDGE_LABELS["ground"]
                )
                c_graph.add_edge(
                    v_of_edge=arg_node, u_of_edge=arg, edge_label=CLG_EDGE_LABELS["ground"]
                )

        return c_graph

    def convert_to_TG(self, G) -> None:
        """
        Converts nx graph into pytorch_geometric tensors and stores them.
        """

        pyg_G = from_networkx(G)
        x = pyg_G.x # 节点特征数据(N * F)

        if self.n_edge_labels == 1:
            edge_indices = pyg_G.edge_index # 边索引数据(2 * E) E为边总数(无向图,边索引为2倍边数量), 2表示连接的两点
        else:
            assert self.n_edge_labels > 1
            edge_indices = [[] for _ in range(self.n_edge_labels)] # 边索引数据(2 * E_l) E_l为标签数
            edge_index_T = pyg_G.edge_index.T
            for i, edge_label in enumerate(pyg_G.edge_label):
                edge_indices[edge_label].append(edge_index_T[i])
            for i in range(self.n_edge_labels):
                if len(edge_indices[i]) > 0:
                    edge_indices[i] = torch.vstack(edge_indices[i]).long().T
                else:
                    edge_indices[i] = torch.tensor([[], []]).long()

        return x, edge_indices
        
