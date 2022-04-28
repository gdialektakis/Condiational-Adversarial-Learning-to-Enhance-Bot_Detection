

def find_percent_of_increase(new_number, old_number):
    increase = new_number-old_number
    perc_of_increase = (increase/old_number)*100

    print('Percentage of increase = {} %'.format(round(perc_of_increase, 2)))



old_num = 0.58011
new_num = 0.63405
find_percent_of_increase(new_num, old_num)