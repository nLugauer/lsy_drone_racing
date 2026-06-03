1. Clean up the code
    * make different classes
    * split up functionallies

2. Implement Obstacles
   * Model Obstacles fast math expression
     * Gates
     * Poles
   * Approach 1: fast online replanning (Qianhao Wang et al.)
     * Uses MINCO Polynomial for trajectory replanning

   * Approach 2: Purely rely on prediction Horizon of MPC with soft Const 
