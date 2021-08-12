### This goal of this project is to generate multiple teams which are both high performant, and fair, using the concept of non dominated sorting. 

#### Note: We have two objectives: Performance and Fairness. We want to maximize both when we form team. Since this is a multi objective optimization problem, we give the pareto optimal set as a result. The users can pick one of these solutions.

##### The meaning of the different versions are as follows:

| Version No. | Functionalities                                                                                               |
|-------------|---------------------------------------------------------------------------------------------------------------|
| Version 1   | Just a vanilla multi-objective team generator with predefined users. We focused on just diversity focused teams |
| Version 2 | Focuses on generating random users, and the solution space was randomized and truncated to improve performance |
| Version 3 | Gives the group admin to individually select formation criteria such as, diversity or similarity for gender, personality, availability each. |
| Version 4 | Same program as version 4, but this parses user data based on API get request |


