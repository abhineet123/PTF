function [ name ] = seq_name( source_id )

dict = 	{'nl_bookI_s3', ...%0
			'nl_bookII_s3', ...%1
			'nl_bookIII_s3', ...%2
			'nl_cereal_s3', ...%3
			'nl_juice_s3', ...%4
			'nl_mugI_s3', ...%5
			'nl_mugII_s3', ...%6
			'nl_mugIII_s3', ...%7
			'nl_bookI_s4', ...%8
			'nl_bookII_s4', ...%9
			'nl_bookIII_s4', ...%10
			'nl_cereal_s4', ...%11
			'nl_juice_s4', ...%12
			'nl_mugI_s4', ...%13
			'nl_mugII_s4', ...%14
			'nl_mugIII_s4', ...%15
			'nl_bus', ...%16
			'nl_highlighting', ...%17
			'nl_letter', ...%18
			'nl_newspaper', ...%19
			'nl_bookI_s1', ...%20
			'nl_bookII_s1', ...%21
			'nl_bookIII_s1', ...%22
			'nl_cereal_s1', ...%23
			'nl_juice_s1', ...%24
			'nl_mugI_s1', ...%25
			'nl_mugII_s1', ...%26
			'nl_mugIII_s1', ...%27
			'nl_bookI_s2', ...%28
			'nl_bookII_s2', ...%29
			'nl_bookIII_s2', ...%30
			'nl_cereal_s2', ...%31
			'nl_juice_s2', ...%32
			'nl_mugI_s2', ...%33
			'nl_mugII_s2', ...%34
			'nl_mugIII_s2', ...%35
			'nl_bookI_s5', ...%36
			'nl_bookII_s5', ...%37
			'nl_bookIII_s5', ...%38
			'nl_cereal_s5', ...%39
			'nl_juice_s5', ...%40
			'nl_mugI_s5', ...%41
			'nl_mugII_s5', ...%42
			'nl_mugIII_s5', ...%43
			'nl_bookI_si', ...%44
			'nl_bookII_si', ...%45
			'nl_cereal_si', ...%46
			'nl_juice_si', ...%47
			'nl_mugI_si', ...%48
			'nl_mugIII_si', ...%49
			'dl_bookI_s3', ...%50
			'dl_bookII_s3', ...%51
			'dl_bookIII_s3', ...%52
			'dl_cereal_s3', ...%53
			'dl_juice_s3', ...%54
			'dl_mugI_s3', ...%55
			'dl_mugII_s3', ...%56
			'dl_mugIII_s3', ...%57
			'dl_bookI_s4', ...%58
			'dl_bookII_s4', ...%59
			'dl_bookIII_s4', ...%60
			'dl_cereal_s4', ...%61
			'dl_juice_s4', ...%62
			'dl_mugI_s4', ...%63
			'dl_mugII_s4', ...%64
			'dl_mugIII_s4', ...%65
			'dl_bus', ...%66
			'dl_highlighting', ...%67
			'dl_letter', ...%68
			'dl_newspaper', ...%69
			'dl_bookI_s1', ...%70
			'dl_bookII_s1', ...%71
			'dl_bookIII_s1', ...%72
			'dl_cereal_s1', ...%73
			'dl_juice_s1', ...%74
			'dl_mugI_s1', ...%75
			'dl_mugII_s1', ...%76
			'dl_mugIII_s1', ...%77
			'dl_bookI_s2', ...%78
			'dl_bookII_s2', ...%79
			'dl_bookIII_s2', ...%80
			'dl_cereal_s2', ...%81
			'dl_juice_s2', ...%82
			'dl_mugI_s2', ...%83
			'dl_mugII_s2', ...%84
			'dl_mugIII_s2', ...%85
			'dl_bookI_s5', ...%86
			'dl_bookII_s5', ...%87
			'dl_bookIII_s5', ...%88
			'dl_cereal_s5', ...%89
			'dl_juice_s5', ...%90
			'dl_mugI_s5', ...%91
			'dl_mugIII_s5', ...%92
			'dl_bookII_si', ...%93
			'dl_cereal_si', ...%94
			'dl_juice_si', ...%95
			'dl_mugI_si', ...%96
			'dl_mugIII_si', ...%97
			'dl_mugII_si', ...%98
			'dl_mugII_s5', ...%99
			'nl_mugII_si', ...%100
			'robot_bookI', ...%101
			'robot_bookII', ...%102
			'robot_bookIII', ...%103
			'robot_cereal', ...%104
			'robot_juice', ...%105
			'robot_mugI', ...%106
			'robot_mugII', ...%107
			'robot_mugIII' }; %108
        
name = dict{source_id+1};
end

