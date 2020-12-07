import numpy as np

status = 0
def payoff(lastPayoff):
        global status
        
        non_empty = 0
        for iteractions in lastPayoff:
            if len(iteractions) > 0:
                non_empty += 1
        if non_empty < 2:
            return 
 
        ## average of last iteration
        last_interation = -1
        for i in range (len(lastPayoff) - 1, -1, -1):
            if len(lastPayoff[i]) > 0:
                last_iteration = i
                break
            
        last_avg = np.mean(lastPayoff[last_iteration])
        
        ## average of second last iteration
        second_last_interation = -1
        for i in range (last_iteration - 1, -1, -1):
            if len(lastPayoff[i]) > 0:
                second_last_interation = i
                break
        
        second_last_avg = np.mean(lastPayoff[second_last_interation])
        
        if last_avg < second_last_avg:
            newStatus = 1 if status == 0 else 0
            print()
            print("last: " + str(last_avg), end = ", ")
            print("second last: " + str(second_last_avg))
        
if __name__ == '__main__':
        
        given = [[-1], [1, -1, 1, 1], [-1, 0, -1, 0, -1, 0], [1, 1, -1, -1, -1, -1, -1], [-1, 0, 1, -1, -1, -1, 0, 0, 0], [0, 0, 0, 0, 2, 2, 2], [0, -1, 0, 1, 0, -1]]
        # print(given)
        
        for i in range (len(given)):
                lastPayoff = given[0:i+1]
                print(lastPayoff)
                payoff(lastPayoff)
