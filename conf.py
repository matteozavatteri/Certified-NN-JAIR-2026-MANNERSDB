# name -> (index,min_range,max_range)
InputFeatures = {
	'mode' : (0,0,1),
	'Number of people' : (1,0,9),
	'Number of people in group' : (2,0,5), 
	'Group radius' : (3,0.5,1),
    'Distance to group' : (4,0,6),
    'Robot within group?' : (5,0,1),
    'Robot facing group?' : (6,0,1),
    'Robot work radius' : (7,0,3),
    'Distance to closest human' : (8,0,6),
    'Distance to 2nd closest human' : (9,0,6),
    'Distance to 3rd closest human' : (10,0,6),
    'Direction to closest human' : (11,0,360),
    'Direction to 2nd closest human' : (12,0,360),
    'Direction to 3rd closest human' : (13,0,360),
    'Direction from closest human to robot' : (14,0,360),
    'Robot facing closest human?' : (15,0,1),
    'Robot facing 2nd closest human?' : (16,0,1),
    'Robot facing 3rd closest human?' : (17,0,1),
    'Closest human facing robot?' : (18,0,1), 
    '2nd closest human facing robot?' : (19,0,1), 
    '3d closest human facing robot?' : (20,0,1), 
    'Number of children' : (21,0,2), 
    'Distance to closest child' : (22,0,6), 
    'Number of animals' : (23,0,1), 
    'Distance to closest animal' : (24,0,6), 
    'Number of people sitting/laying in sofa?' : (25,0,2), 
    'Music playing?' : (26,0,1), 
    'Number of agents in scene' : (27,1,11)
}

# name -> (index,min_range,max_range)
OutputScores = {
	'Vacuum cleaning' : (0,1,5),
	'Mopping the floor' : (1,1,5),
	'Carry warm food' : (2,1,5),
	'Carry cold food' : (3,1,5),
	'Carry drinks' : (4,1,5),
    'Carry small objects (plates, toys)' : (5,1,5),
    'Carry big objects (tables, chairs)' : (6,1,5),
    'Cleaning (Picking up stuff) / Starting conversation' : (7,1,5)
}

InputSIZE = 28
OutputSIZE = 8

BinaryIndexes = [0, 5, 6, 15, 16, 17, 18, 19, 20, 23, 26]
ContinuousIndexes = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 24, 25, 27]

index_dtch = 8
index_dtcc = 22
index_dtca = 24

index_vacuum    = 0
index_mop       = 1
index_warm_food = 2
index_cold_food = 3
index_drinks    = 4
index_small_obj = 5
index_big_obj   = 6
index_clean_conversation = 7

ActionsDifferentFromMop         = [act for act in range(OutputSIZE) if act != index_mop]
ActionsDifferentFromWarmFood    = [act for act in range(OutputSIZE) if act != index_warm_food]
ActionsDifferentFromCarryBigObj = [act for act in range(OutputSIZE) if act != index_big_obj]
