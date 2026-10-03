class ReasoningSolver:
    @staticmethod
    def solve_direct_comparison(relations, question_type="max"):
        """
        relations: list of tuples (Entity1, operator, Entity2). operator can be '>' or '<'
        question_type: "max" or "min"
        """
        greater_than = {}
        entities = set()
        for e1, op, e2 in relations:
            entities.add(e1)
            entities.add(e2)
            if e1 not in greater_than: greater_than[e1] = set()
            if e2 not in greater_than: greater_than[e2] = set()
            
            if op == '>':
                greater_than[e1].add(e2)
            elif op == '<':
                greater_than[e2].add(e1)
                
        # In a directed graph where u -> v means u is greater than v:
        # The greatest entity has in-degree 0 (nobody is greater than it).
        # The smallest entity has out-degree 0 (it is not greater than anybody).
        in_degree = {e: 0 for e in entities}
        for u in greater_than:
            for v in greater_than[u]:
                in_degree[v] += 1
                
        if question_type == "max":
            candidates = [e for e in entities if in_degree[e] == 0]
            # Ensure unique answer and graph connectivity (if queried, it shouldn't be disconnected)
            return candidates[0] if len(candidates) == 1 else None
        elif question_type == "min":
            candidates = [e for e in entities if len(greater_than[e]) == 0]
            return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def solve_attribute_comparison(attributes, question_type="max"):
        """
        attributes: dict {Entity: value}
        """
        if not attributes: return None
        if question_type == "max":
            max_val = max(attributes.values())
            candidates = [e for e, v in attributes.items() if v == max_val]
            return candidates[0] if len(candidates) == 1 else None
        elif question_type == "min":
            min_val = min(attributes.values())
            candidates = [e for e, v in attributes.items() if v == min_val]
            return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def solve_transitive(relations, e1, e2):
        """
        Determines the relation between e1 and e2.
        Returns '>', '<', '=', or '?'
        """
        greater_than = {}
        equal_to = {}
        entities = set()
        
        for u, op, v in relations:
            entities.add(u); entities.add(v)
            if u not in greater_than: greater_than[u] = set()
            if v not in greater_than: greater_than[v] = set()
            if u not in equal_to: equal_to[u] = set()
            if v not in equal_to: equal_to[v] = set()
            
            if op == '>': greater_than[u].add(v)
            elif op == '<': greater_than[v].add(u)
            elif op == '=':
                equal_to[u].add(v)
                equal_to[v].add(u)
        
        def check_greater(start, target):
            visited = set()
            queue = [(start, False)] # (node, has_strict_greater)
            while queue:
                curr, has_greater = queue.pop(0)
                if curr == target and has_greater: return True
                if curr in visited: continue
                visited.add(curr)
                
                for nxt in greater_than.get(curr, []):
                    queue.append((nxt, True))
                for nxt in equal_to.get(curr, []):
                    queue.append((nxt, has_greater))
            return False

        if check_greater(e1, e2): return '>'
        if check_greater(e2, e1): return '<'
        
        def check_equal(start, target):
            visited = set()
            queue = [start]
            while queue:
                curr = queue.pop(0)
                if curr == target: return True
                if curr in visited: continue
                visited.add(curr)
                for nxt in equal_to.get(curr, []):
                    queue.append(nxt)
            return False
            
        if check_equal(e1, e2): return '='
        return '?'

# Test suite as requested
if __name__ == "__main__":
    s = ReasoningSolver()
    
    # 1. Direct Comparison specific tests
    # A > B, B > C
    assert s.solve_direct_comparison([('A', '>', 'B'), ('B', '>', 'C')], 'max') == 'A'
    assert s.solve_direct_comparison([('A', '>', 'B'), ('B', '>', 'C')], 'min') == 'C'
    
    # A < B, B < C
    assert s.solve_direct_comparison([('A', '<', 'B'), ('B', '<', 'C')], 'max') == 'C'
    assert s.solve_direct_comparison([('A', '<', 'B'), ('B', '<', 'C')], 'min') == 'A'
    
    # A > B, A > C, B > C
    assert s.solve_direct_comparison([('A', '>', 'B'), ('A', '>', 'C'), ('B', '>', 'C')], 'max') == 'A'
    assert s.solve_direct_comparison([('A', '>', 'B'), ('A', '>', 'C'), ('B', '>', 'C')], 'min') == 'C'
    
    # Disconnected/Ambiguous graph
    assert s.solve_direct_comparison([('A', '>', 'B'), ('C', '>', 'D')], 'max') == None
    assert s.solve_direct_comparison([('A', '>', 'B'), ('C', '>', 'B')], 'max') == None # A and C are both max
    
    # 2. Multi-hop Logic specific tests
    # A > B, B > C, C > D -> A and D
    assert s.solve_transitive([('A', '>', 'B'), ('B', '>', 'C'), ('C', '>', 'D')], 'A', 'D') == '>'
    
    # A > B, C < B -> A and C
    assert s.solve_transitive([('A', '>', 'B'), ('C', '<', 'B')], 'A', 'C') == '>'
    
    # A > B, B = C -> A and C
    assert s.solve_transitive([('A', '>', 'B'), ('B', '=', 'C')], 'A', 'C') == '>'
    
    # A > B -> B and A
    assert s.solve_transitive([('A', '>', 'B')], 'B', 'A') == '<'
    
    print("All logic solver tests passed!")
