#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[Directory[]]


(* ::Section:: *)
(*Read input:*)


(* Read the input file and define global variables *)
infilename = $ScriptCommandLine[[2]]
ndims  = StringSplit[Find[infilename, "dimensions :"], ":"][[-1]]//ToExpression
potfile = StringSplit[Find[infilename, "potential :"], ":"][[-1]]//ToExpression
dt = StringSplit[Find[infilename, "dt"], ":"][[-1]]//ToExpression
stride = StringSplit[Find[infilename, "stride"], ":"][[-1]]//ToExpression


potentialNC = Import[potfile, "Data"];
wffileNC = Import["WF.nc", "Data"];


(* ::Section:: *)
(*Function section:*)


CombinePlots = ResourceFunction["CombinePlots"]

Plot2D[potfile_, wffile_] := 
 Module[{potential, pAmp, potPlot, barleg, pAmpPlot, grid},
  potential = 
   Flatten[Table[{potfile["/xdim"][[x]], potfile["/ydim"][[y]], 
      potfile["/potential"][[x, y]]*27.11}, 
    {x, 1, Length[potfile["/xdim"]], 10}, {y, 1, Length[potfile["/ydim"]], 10}], 1];
  barleg = 
   BarLegend[{ColorData["TemperatureMap", "ColorFunction"][#/Max[potential[[;; , 3]]]-Min[potential[[;; , 3]]]]&, 
    {Min[potential[[;; , 3]]], Max[potential[[;; , 3]]]}},
    Frame -> True,
    LabelStyle -> Directive[Black, 12, FontFamily -> "Helvetica"],
    TicksStyle -> Directive[FontFamily -> "Helvetica", 12, Black], 
    FrameStyle -> Directive[Black, AbsoluteThickness[1.2]],
    LegendMarkerSize -> {15, 180},
    LegendLabel -> Placed["E [eV]", Top]];
  potPlot = ListDensityPlot[potential,
    ColorFunction -> "TemperatureMap",
    MaxPlotPoints -> 10];
  grid = Table[
    pAmp = 
     Flatten[Table[{wffile["/Xdim"][[x]], wffile["/Ydim"][[y]], 
        wffile["/PAmp"][[Round[t], x, y]]}, 
    {x, 1, Length[wffile["/Xdim"]], 10}, {y, 1, Length[wffile["/Ydim"]], 10}], 1];
    pAmpPlot = ListContourPlot[pAmp,
      PlotRange -> All,
      ContourShading -> None,
      ContourStyle -> Directive[Black, Thick],
      Contours -> 10];
    Labeled[
     Show[{potPlot, pAmpPlot}, AspectRatio -> 1, 
      FrameStyle -> Directive[Black, 12, FontFamily -> "Helvetica", 
        AbsoluteThickness[1.5]],
      PlotRangePadding -> None,
      GridLines -> Automatic,
      GridLinesStyle -> Dashed,
      ImageSize -> 200],
     Flatten@{Style[#, 12, Black, 
          FontFamily -> "Helvetica"] & /@ {"X [Bohrs]", "Y [Bohrs]", 
         ToString[NumberForm[t*dt*stride*10/41.3413, {5, 1}]] <> " fs"}, 
       Rotate[barleg, -Pi/2]},
     {Bottom, Left, Top, Right}, RotateLabel -> True, 
     Spacings -> {0.2, 0.2}],
    {t, 1, Dimensions[wffile["/PAmp"]][[1]], Dimensions[wffile["/PAmp"]][[1]]/8}];
  grid = Grid[ArrayReshape[grid, {2, 4}]];
  Export["PAmp_evol.pdf", grid];
  Export["PAmp_evol.png", Rasterize[grid, ImageResolution -> 400]];
  ]

Plot1D[potfile_, wffile_] :=  
  Module[{potential, pAmp, potPlot, times, pAmpPlot, grid, cols},
   potential = 
    Table[{potfile["/xdim"][[i]], 
      potfile["/potential"][[i]]*27.11}, 
    {i, Length[potfile["/xdim"]]}
    ];
   cols = ColorData[97, "ColorList"][[;; 8]];
   times = Table[
     ToString[NumberForm[t*dt*stride*10/41.3413, {5, 1}]] <> " fs",
     {t, 1, Dimensions[wffile["/PAmp"]][[1]],Dimensions[wffile["/PAmp"]][[1]]/8}];
   pAmp = Table[
     Table[{wffile["/Xdim"][[i]], wffile["/PAmp"][[Round[t], i]]}, {i,
        Length[wffile["/Xdim"]]}],
     {t, 1, Dimensions[wffile["/PAmp"]][[1]],Dimensions[wffile["/PAmp"]][[1]]/8}
    ];
   grid = Table[Labeled[
      CombinePlots[
       ListLinePlot[potential,
        Frame -> True,
        PlotStyle -> Directive[Gray, AbsoluteThickness[2], Dashed], 
        PlotRange -> All],
       ListLinePlot[pAmp[[i]],
        PlotRange -> All,
        Frame -> True,
        PlotStyle -> cols[[i]]],
       FrameStyle -> Directive[Black, 11, AbsoluteThickness[1.2]],
       Axes -> False,
       ImagePadding -> Automatic,
       ImageSize -> 240,
       "AxesSides" -> "TwoY"],    	   
      Style[#, 11, Black, FontFamily -> "Helvetica"] & /@ {"X [Bohr]",
         "E [eV]", times[[i]], "Prob. Amp."}, {Bottom, Left, Top, 
       Right}, RotateLabel -> True, Spacings -> {.4, .4}], {i, 1, 8}];
     grid = Grid[ArrayReshape[grid, {2, 4}]];
     Export["PAmp_evol.pdf", grid];
     Export["PAmp_evol.png", 
    Rasterize[grid, ImageResolution -> 400]];
   ];


(* ::Section:: *)
(*Main code*)

(*Print[
"#============================================
            Quantum Dynamics Engine
=============================================#"]*)

If[ ndims == 1,
    Plot1D[potentialNC, wffileNC],
    If[ ndims == 2,
        Plot2D[potentialNC, wffileNC]]
    ]

