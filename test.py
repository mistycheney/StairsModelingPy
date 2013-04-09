#!/bin/python

def fact(n):
	print n
    if n == 1:
    	return 1
    else:
    	return n*fact(n-1)

def reverse(s_str):
	s = []
	s.extend(s_str)
	for i in range(len(s)/2):
	    if i == len(s)-i-1:
	        continue
	    temp = s[i]
	    s[i] = s[-i-1]
	    s[-i-1] = temp
	return s

import random


def dice(n):
    count = [0]*6
    for i in range(n):
        v = random.randint(1,6)
        print 'the', i, 'th die is', v
        count[v-1] += 1
    # now we calculate the probability of this combination
    # based on count. This is a multinomial distribution.
    
    denom = reduce(lambda x, c: x*fact(c), count, 1)
    p = fact(n)/denom*(float(1)/6)**n
    return p

if __name__ == '__main__':
    # print fact(0)
    print dice(3)
