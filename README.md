# rl-easy21
My solutions for David Silver's UCL course on Reinforcement Learning. http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html

## Discussion

### What are the pros and cons of bootstrapping in Easy21?
TD-learning has smaller variance but might be biases. Comparing it to the unbiased estimation of the MC method using MSE, we can see that the estimated Q function is not far off and while it converged much faster.

### Would you expect bootstrapping to help more in blackjack or Easy21 ?Why?
In blackjack we have an Ace which might make an episode a bit longer. Easy21 has red cards which also make the episodes longer. 
It is not straightforward, but I believe on average, Easy21 episodes will be longer (considering the chances of getting an Ace or a red card).
As the episodes will be longer, bootstrapping which has less variance will help it more. 

### What are the pros and cons of function approximation in Easy21?
Faster convergence (as we have less parameters) but worse approximation.

### How would you modify the function approximator suggested in this section
to get better results in Easy21?
As the total return is {-1,0,1} we can add a tanh layer that will make sure the output is within these limits.
