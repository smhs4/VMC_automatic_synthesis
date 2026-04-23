Break down the search:

- investigate damping and how to improve it
exploring the damping effect, joint damping just reduces everything and takes energy out of the system while damping the srping itself causes behaviour such as moving in a direction tangent to the initial velocity.
[ ] need to generate data for this

one arm and a fixed point: high damping up to (30N/m and 10N/m/s) create a stable movement and is necassary for good motion.

- Jitter causes instability (Maybe can create some results and plot them)

- investigate connection to ground (probably igonore)

- improve robustness this is done in simulation


- investigate good grip (this is done by applying a small lift )

- investigate good lift

- [x] camera lag (does not seem to be a problem)

- visualisation ()
- stream line upload process
- [x] initial pose 

- need to investigate why many springs cause high ausilations while small forces do not cause high ausilations

- what vmc setup will lead to a robust grip



Completed this week:

- Initial pose set up using pid and tuned parameters
- Single spring high stiffness and damping 


- camera lag was investigated and does not seem to be the primary issue (for one spring) might become an issue with multiple. -- camera lag might be the fundemental source.

- add damping to the joint
- penalise tendon imbalance


- try non-linear damping

- adjust for the variable interval time

- is velocity noisy

- try by process of elimination

Needed for next week:

we need a table of content of what we have done. and one more chapter before.


Fighting friction:
- Adding jitter
- Initial high force to get it moving
- Damping can be position dependent or more related to the velocity like squared…

Adding some posture control

Adding padding to the object and the chopsticks

Use other cameras.

Try to learn in the real world

Tune the damping

Try on a bigger box

Getting to starting position automatically

Add damping 

Limit forces in mujoco ( 1kg force )

Smoothing the velocity



---------------------------

results 22 Apr:

- velocity filters do not work
- 
