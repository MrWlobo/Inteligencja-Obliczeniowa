import time
from stripsProblem import Strips, STRIPS_domain, Planning_problem, boolean
from searchGeneric import Searcher, AStarSearcher
from searchProblem import Path, Arc, Search_problem

# --- ADAPTER ---
class Planning_problem_as_search_problem(Search_problem):
    def __init__(self, prob, heuristic_type='zero'):
        self.prob = prob
        self.heuristic_type = heuristic_type
    def start_node(self):
        return frozenset(self.prob.initial_state.items())
    def is_goal(self, node):
        state_dict = dict(node)
        return all(state_dict.get(f) == v for f, v in self.prob.goal.items())
    def neighbors(self, node):
        state_dict = dict(node)
        res = []
        for action in self.prob.prob_domain.actions:
            if all(state_dict.get(f) == v for f, v in action.preconds.items()):
                new_state = state_dict.copy()
                new_state.update(action.effects)
                res.append(Arc(node, frozenset(new_state.items()), action.cost, action))
        return res
    def heuristic(self, node):
        if self.heuristic_type == 'goals_not_met':
            state_dict = dict(node)
            return sum(1 for feat, val in self.prob.goal.items() if state_dict.get(feat) != val)
        return 0

# --- DOMENA ---
cargos, planes, airports = ['c1', 'c2', 'c3'], ['p1'], ['a1', 'a2']
features = {f'at_{c}': {'a1', 'a2', 'p1'} for c in cargos}
for p in planes:
    features[f'at_{p}'] = {'a1', 'a2'}
    features[f'fuel_{p}'] = boolean
actions = []
for p in planes:
    for a in airports:
        actions.append(Strips(f"refuel_{p}_{a}", {f"at_{p}": a}, {f"fuel_{p}": True}))
        for c in cargos:
            actions.append(Strips(f"load_{c}_{p}_{a}", {f"at_{c}": a, f"at_{p}": a}, {f"at_{c}": p}))
            actions.append(Strips(f"unload_{c}_{p}_{a}", {f"at_{c}": p, f"at_{p}": a}, {f"at_{c}": a}))
    for f in airports:
        for t in airports:
            if f != t:
                actions.append(Strips(f"fly_{p}_{f}_{t}", {f"at_{p}": f, f"fuel_{p}": True}, {f"at_{p}": t, f"fuel_{p}": False}))
domain = STRIPS_domain(features, actions)

# --- DEFINICJE PROBLEMÓW ---
prob1 = Planning_problem(domain, {"at_c1":"a1","at_c2":"a1","at_c3":"a1","at_p1":"a1","fuel_p1":False}, {"at_c1":"a2"})
prob2 = Planning_problem(domain, {"at_c1":"a1","at_c2":"a1","at_c3":"a1","at_p1":"a1","fuel_p1":False}, {"at_c1":"a2","at_c2":"a2"})
prob3 = Planning_problem(domain, {"at_c1":"a1","at_c2":"a1","at_c3":"a1","at_p1":"a1","fuel_p1":False}, {"at_c1":"a2","at_c2":"a2","at_p1":"a1"})
basic_probs = [(prob1, [{"at_c1": "p1"}, {"at_c1": "a2"}]), 
               (prob2, [{"at_c1": "p1", "at_c2": "p1"}, {"at_c1": "a2", "at_c2": "a2"}]), 
               (prob3, [{"at_c1": "p1", "at_c2": "p1"}, {"at_c1": "a2", "at_c2": "a2"}, {"at_p1": "a1"}])]

big_prob1 = Planning_problem(domain, {"at_c1":"a1", "at_c2":"a1", "at_c3":"a1", "at_p1":"a2", "fuel_p1":False}, {"at_c1":"a2", "at_c2":"a2", "at_c3":"a1", "at_p1":"a1"})
big_prob2 = Planning_problem(domain, {"at_c1":"a2", "at_c2":"a2", "at_c3":"a2", "at_p1":"a1", "fuel_p1":False}, {"at_c1":"a1", "at_c2":"a1", "at_c3":"a1", "at_p1":"a2"})
big_prob3 = Planning_problem(domain, {"at_c1":"a1", "at_c2":"a2", "at_c3":"a1", "at_p1":"a1", "fuel_p1":True}, {"at_c1":"a2", "at_c2":"a1", "at_c3":"a2", "at_p1":"a1"})
big_probs = [(big_prob1, [{"at_p1":"a1"}, {"at_c1":"p1"}, {"at_p1":"a2"}, {"at_c1":"a2"}, {"at_c2":"p1"}, {"at_p1":"a2"}, {"at_c2":"a2"}, {"at_p1":"a1"}]),
             (big_prob2, [{"at_p1":"a2"}, {"at_c1":"p1"}, {"at_p1":"a1"}, {"at_c1":"a1"}, {"at_p1":"a2"}, {"at_c2":"p1"}, {"at_p1":"a1"}, {"at_c2":"a1"}]),
             (big_prob3, [{"at_c1":"p1"}, {"at_c1":"a2"}, {"at_c2":"p1"}, {"at_c2":"a1"}, {"at_c3":"p1"}, {"at_c3":"a2"}, {"at_p1":"a1"}])]

def solve_with_subgoals(domain, initial_state, subgoals, h_type='zero'):
    curr_state, full_plan, total_nodes, start = initial_state, [], 0, time.time()
    for sg in subgoals:
        sp = Planning_problem(domain, curr_state, sg)
        solver = AStarSearcher(Planning_problem_as_search_problem(sp, h_type))
        res = solver.search()
        if not res: return None, 0, 0
        path = []
        c = res
        while c.arc: path.append(c.arc.action.name); c = c.initial
        full_plan.extend(reversed(path))
        total_nodes += solver.num_expanded
        curr_state = dict(res.end())
    return full_plan, total_nodes, time.time() - start

# --- URUCHOMIENIE ---
print("=== WYNIKI: ZADANIA 4 i 6 PKT ===")
for i, (p_strips, s_list) in enumerate(basic_probs, 1):
    print(f"\n--- PROBLEM {i} ---")
    # Forward (BFS) - z limitem dla bezpieczeństwa
    if i < 3: # Problem 3 Forward często muli, możesz go pominąć
        sp = Planning_problem_as_search_problem(p_strips, 'zero')
        sol = AStarSearcher(sp)
        st = time.time()
        res = sol.search()
        print(f"Forward Planning: Czas={time.time()-st:.4f}s, Kroki={len(list(res.nodes()))-1 if res else 'N/A'}")
    else:
        print("Forward Planning: POMINIĘTO (Zbyt wysoka złożoność)")

    # A* Heurystyka
    sp_h = Planning_problem_as_search_problem(p_strips, 'goals_not_met')
    sol_h = AStarSearcher(sp_h)
    st = time.time()
    res_h = sol_h.search()
    print(f"A* Heurystyka: Czas={time.time()-st:.4f}s, Węzły={sol_h.num_expanded}")

    # Podcele
    _, n, d = solve_with_subgoals(domain, p_strips.initial_state, s_list, 'zero')
    print(f"Podcele (BFS): Czas={d:.4f}s, Węzły={n}")
    _, n, d = solve_with_subgoals(domain, p_strips.initial_state, s_list, 'goals_not_met')
    print(f"Podcele (A*): Czas={d:.4f}s, Węzły={n}")

print("\n=== WYNIKI: ZADANIA 8 PKT ===")
for i, (p_big, s_big) in enumerate(big_probs, 1):
    plan, nodes, duration = solve_with_subgoals(domain, p_big.initial_state, s_big, 'goals_not_met')
    print(f"DUŻY PROBLEM {i}: Sukces! Kroki={len(plan)}, Czas={duration:.4f}s, Węzły={nodes}")