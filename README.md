## Max Flow Algorithms: Ford-Fulkerson, Edmonds-Karp & Push-Relabel (Serial vs Parallel)

🔗 [Read Full Article on Medium](https://medium.com/@pranay-reddy-palle/max-flow-algorithms-ford-fulkerson-edmonds-karp-push-relabel-serial-vs-parallel-xyz)

Paper Link Soon

Flow networks power countless real-world systems — from airline scheduling and internet routing to supply chains and medical image segmentation. This article dives into three key max-flow algorithms with a special emphasis on parallelization:

### 📌 Algorithms Covered

- **Ford-Fulkerson**: Repeatedly finds augmenting paths using DFS.
  - ⏱️ Time Complexity: `O(max_flow × E)`
- **Edmonds-Karp**: A BFS-based implementation of Ford-Fulkerson for better performance.
  - ⏱️ Time Complexity: `O(V × E²)`
- **Push-Relabel (Preflow-Push)**: Uses local flow pushing and node relabeling.
  - ⏱️ Time Complexity (Serial): `O(V² × E)`  
  - ⚡ **Parallel Variant**: Up to 6x speedup using atomic operations and concurrent updates on GPUs.

### ⚖️ Serial vs Parallel Push-Relabel

| Variant       | Approach                          | Performance Boost |
|---------------|-----------------------------------|-------------------|
| **Serial**    | One active node at a time         | —                 |
| **Parallel**  | Many active nodes in parallel     | 🔼 Up to 6x speedup |

### Real-World Use Cases

- **Telecom Routing** → Edmonds-Karp for consistent throughput
- **Medical Image Segmentation** → Push-Relabel for fast 3D/4D processing
- **Supply Chains & Energy Grids** → Ford-Fulkerson for quick reconfigurations

### 📚 Read More
- Article: [Max Flow Breakdown](https://medium.com/@pranay-reddy-palle/max-flow-algorithms-ford-fulkerson-edmonds-karp-push-relabel-serial-vs-parallel-xyz)
- [Josh Korn's Flow Algorithm Guide](https://joshkorn.com/flows.html)
- Stay tuned for my upcoming paper on **Parallel Push-Relabel using CUDA** 🧠


