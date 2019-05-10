from lib import astar, jps, astar_obj, jps_obj
from time import time as timer


def search(grid, type, prevLine, line, startX, endX):
    

    #A* takes steps of 2, so if everything is even it's safer
    prevLine = 2*(prevLine // 2)
    line = 2*(line // 2)

    begin_search = timer()

    start, goal = get_start_and_goal(prevLine, line, startX, endX)
    # start, goal = [line, 0], [line + 20, 3320]

    if type == 'A':
        print 'A*..'
        a = astar.Astar(grid)
        path, map = a.pathfind(start, goal)

    elif type == 'jps':
        print 'A* + JPS...'
        j = jps.Jps(grid)
        path, map = j.pathfind(start, goal)

    print ' => path found in ' + str(timer() - begin_search) + ' s'

    return path, map


def get_start_and_goal(p, l, startX, endX):
    return [p, startX], [l, endX]
