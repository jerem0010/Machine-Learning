from collections import deque
from typing import Optional, List

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_max = float('-inf')
            level_size = len(queue)

            for _ in range(level_size):
                node = queue.popleft()
                level_max = max(level_max, node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(level_max)

        return result

# Exemple d'utilisation avec l'arbre [1, 3, 2, 5, 3, None, 9]
tree = TreeNode(1, TreeNode(3, TreeNode(5), TreeNode(3)), TreeNode(2, None, TreeNode(9)))

# Création d'une instance de Solution
solution = Solution()

# Appel de la méthode largestValues avec l'arbre que nous avons créé
result = solution.largestValues(tree)

# Affichage du résultat
print(result)