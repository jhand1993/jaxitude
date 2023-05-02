# attitude

An intuitive attitude calculations and conversions in Python written with JAX.

## Known Issues

-[ ] compose_MRP: shadow conversion not working.
-[X] get_prv: there might be a bug here.  Need to test.  Looks good
-[X] Something is wrong with Euler angles.  I think it has to do with definitions from wikipedia (active?) versus book (passive?).  Yup, had rotation types mixed up. Book is passive.
-[ ] If I take three Euler angles, get a dcm, and then get the Euler angles back, the result is the negation of the starting angle set. 
