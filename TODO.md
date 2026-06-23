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




Problem with trajectory planner:
* sometimes exits the allowed area (room boundaries in toml)
* slow startup, I think its coming from mpcc
* no obstacle detection for gate-stand, some nasty tracks will fly under
* spline artifacts when replanning
* whats v_theta?:
   ```
           yref_target[8] = (
            5.0  # Target progress speed (v_theta), matches state constraint upper bound
        )
   ```

* ~~does often get stuck at obstacles: gradient better~~
  * already does this inside of the obstacle
  * if any changes here switch to MPCC++ or gradiant flieds
* 

* if gates are close to each other the replanning is happening to fast, leaving the drone all by its own

* jump in cost funtion -> instability, look picture