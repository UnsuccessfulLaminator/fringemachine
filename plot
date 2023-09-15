stats 'data' using 1:3 nooutput name 'xdata_'
stats 'data' using 2:3 nooutput name 'ydata_'

set hidden3d
set dgrid3d 10,10 qnorm 1
set xlabel '1st voltage'
set ylabel '2nd voltage'
set zlabel 'Phase unevenness' rotate
set object circle at xdata_pos_min_y,ydata_pos_min_y,xdata_min_y

splot 'data' with lines

pause mouse close
