# MultiAgent-Reinforcement-Learning

The purpose of the project is to train multiple RL agents in the same env with a common goal.
The agents get a better reward on communicating and working together on the same target at a time.
Individually it is impossible to solve the env for an agent

Environment – “CityOnFire”:
----------------------------
Square Grid World of adaptive sizes – models a city scape
Each tiny square can be though of as a city block.
No of fires: min:1, max: no limit defined (50 is a good extreme)
Each fire is defined as an object with location and heat
A global counter numbers each fire’s ID
All fires spawn with heat == 100.0
Fires start at random locations with every episode.


Stochasticity:
--------------
Each fire in the grid will increase its heat stochastically by 2.5 or 5.0 degrees with every step of the environment.
Initialization is always random.


FireTrucks are simulated as agents.
Agent Definition:
 -Maximum of 4 Agents can be initiated.
 -Each agent will be initiated at any one corner
 -Each agent knows its status: Active, Engaged, Reached
 -Each agent knows its target: Fire_ID, Loc, if fire is off
 
 Action Space:
 -The degree of movements is 4: 
    Up: 0, Down: 1, Left: 2, Right: 3
 -The degree of Actions possible to take is 4:
    Action 0: Move (up / down / left / right)
    Action 4: Douse Fire
    Action 5: Call For Help
    Action 6: Dis-engage
  Calling for help creates a new message object and stores it in the environment.

Agent Action Policy:
----------------------

Is defined by a DQN named Act_Net
Takes the Action state as input vector
Softmax output of 4 different actions
Training uses Actions Buffer of previous states
Target Act network is used as in DQN

Action Movement Policy:
-----------------------

Is defined by a DQN named Move_Net
Takes the position state as input vector (coordinates)
Softmax output of 4 different actions
Training uses Move Buffer of previous states
Target Move network is used as in DQN

Reward Definiton:
-----------------
Moving Closer towards targets: +0.5
Moving away or staying in same position: -1.0
Reaching a Target: +2.0
Not using water jet at target: -2.0
Using water jet at target: +1
Calling for unnecessary help: -2.0
Calling help when not capable: +5.0
Fire is off at a location: +10 to all agents whose target is that fire



