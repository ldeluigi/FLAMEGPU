 set key title "States"
 set xlabel "Iteration number"
 set ylabel "Population count"
 set key box opaque
 set border back
 
 set terminal pngcairo size 1280,720
 set output "graph.png"
 
 plot 'out' using 1:2 with lines lc 'cyan' title 'too close', 'out' using 1:3 with lines lc 'yellow' title 'aligning', 'out' using 1:4 with lines lc 'magenta' title 'alone'
 
 set output
 set terminal pop