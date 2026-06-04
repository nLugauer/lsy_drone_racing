1. [x] Clean up the code
    * make different classes
    * split up functionalities

2. [ ] Implement Obstacles
   * Improve Obstacle model from cylinder to smth. continuous
   
3. [ ] Implement Track online replanning:
   * minimal snap
   * MINCO
   * > 20ms max for 50Hz

4. [ ] Robustness
   * Tube MPC: Radius of uncertainty + Perception has to be the Terminal / Feasible Set
   * make some literature research about that
   * come up with expression and proof
   * Downside: could be to conservative...
     * -> is there a thin like stochastic feasible sets for tuning conservatism with aggressiveness? 
