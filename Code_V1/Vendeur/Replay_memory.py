from collections import namedtuple
import random

Transition = namedtuple(
        'Transition',
        ('state', 'action', 'next_state', 'reward'))

e = Transition(2,3,1,4) 

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        self.experience = namedtuple('Experience',
        ('state', 'action', 'next_state', 'reward'))
        
    'store experience in ReplayMemory when an experience occur'
    def push(self, state, action, next_state, reward):
        e = self.experience(state, action, next_state, reward)
        if len(self.memory) < self.capacity:
            self.memory.append(e)
        else:
            self.memory[self.push_count % self.capacity] = e
        self.push_count += 1
    
    'sample N random experiences, N=batch_size'
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    "return bool whether it's possible to sample from memory"
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
   
    
'''test
memory=ReplayMemory(100)

memory.push(2,1,4,1)
memory.push(1,0,4,1)
memory.push(4,1,3,-1)
memory.push(1,1,2,1)
memory.push(3,1,1,-1)
memory.push(3,0,4,1)
memory.sample(3)

memory.sample(4)
'''