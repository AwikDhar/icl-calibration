from typing import Dict, List, Optional
import numpy as np
import torch

class TrieNode:
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.logit: float = None
        self.labels: List[str] = []

class LabelsTrie:
    def __init__(self, label_map: Dict[str, List[str]]):
        self.root = TrieNode()
        self.label_map = label_map
        self._build_trie()
    
    def _build_trie(self):
        for label_dict in self.label_map.values():
            node = self.root
            for token in label_dict['tokens']:
                if token not in node.children:
                    node.children[token] = TrieNode()
                node = node.children[token]
            node.labels.append(label_dict['label'])
    
    def update_logits(self, path: List[str], token_logits: Dict[str, float]):
        node = self.root
        for token in path:
            node = node.children[token]
        for token in node.children:
            # node.children[token].logit = token_logits.get(token, float("-inf"))
            token_variants = [
                    token,                    # exact match
                    token.capitalize(),       # capitalized
                    # token.lstrip(),          # without leading space
                    # token.lstrip().capitalize(), # without space + capitalized
                    # ' ' + token,             # with leading space
                    # ' ' + token.capitalize() # with space + capitalized
            ]
            token_variants = list(set(token_variants))
            
            # 64 bit precision is important to prevent overflow(inf)
            logit_variants = torch.tensor([token_logits.get(tk, float('-inf')) for tk in token_variants], dtype=torch.float64) 
            logit = torch.logsumexp(logit_variants, dim=0).cpu().numpy()
            # if token.lstrip() in token_logits:
            #     logit = token_logits[token.lstrip()]
            # elif token in token_logits:
            #     logit = token_logits[token]
            # elif token.capitalize() in token_logits:
            #     logit = token_logits[token.]
            # else:
            #     logit = float("-inf")
            # logit = max(token_logits.get(token, float("-inf")), 
            #             token_logits.get(token.capitalize(), float("-inf")))
            node.children[token].logit = logit
        # if node==self.root and np.sum(np.exp([child.logit for child in node.children.values()]))==0:
        #     print(token_logits); exit()
        
    def get_label_prob(self, tokens: List[str]) -> float:
        node = self.root
        probs = []
        
        # print(tokens)
        for token in tokens:
            logits = [child.logit for child in node.children.values()]
            token_logit = node.children[token].logit
            
            exp_logits = np.exp(logits)
            if np.all(np.isneginf(logits)):
                print(f"No logits for these tokens : {list(node.children.keys())}")
                return 0.0
            
            if np.inf in exp_logits:
                print(f"Inf exp(logit) for tokens: {list(node.children.keys())} and logits: {logits}")
            prob = np.exp(token_logit) / np.sum(exp_logits)
            
            probs.append(prob)
            node = node.children[token]
            
        return np.prod(probs)
    
    def get_all_label_probs(self) -> Dict[str, float]:
        return np.array([self.get_label_prob(self.label_map[label_idx]['tokens']) 
                for label_idx in range(len(self.label_map))])
        
    def print_trie(self, node: Optional[TrieNode] = None, prefix: str = "", token: str = "ROOT"):
        """Print the trie structure for debugging."""
        if node is None:
            node = self.root
        
        logit_str = f" (logit: {node.logit:.2f})" if node.logit is not None else ""
        end_str = f" [END: {', '.join(node.labels)}]" if node.labels else ""
        print(f"{prefix}{token}{logit_str}{end_str}")
        
        for child_token, child_node in node.children.items():
            self.print_trie(child_node, prefix + "  ", child_token)
        
