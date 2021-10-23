# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:11:46 2019
@author: Jonathan
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import pathlib
import pywt
from scipy import optimize

from pylab import rcParams

'''Setting the global figure size'''
rcParams['figure.figsize'] = 10, 10

def analyzeFile(filepath):
    print(f'Processing file: {filepath}')
    filename, extension = os.path.splitext(os.path.basename(filepath))

    '''Creating an output directory'''
    outDir = f'{os.path.dirname(filepath)}/out'
    pathlib.Path(outDir).mkdir(parents=True, exist_ok=True) 


    '''strip empty lines from the input file as this seems to prevent parsing when using non-whitespace delimiters with genfromtxt'''
    strippedPath = f'{outDir}/{filename}_stripped{extension}';
    with open(filepath) as infile, open(strippedPath, 'w') as outfile:
        for line in infile:
            if not line.strip(): continue #skip
            outfile.write(line)

    data = np.genfromtxt(strippedPath,dtype='str',delimiter=",", skip_header=2) if (extension == '.csv') else np.genfromtxt(strippedPath,dtype='str')

    dates = data[:,1]
    dateBins = np.unique(dates)
    maxBins = len(dateBins)*48
    dBmin = []
    dBmax = []

    for i in np.arange(0,len(dateBins)):
        dBmin.append(np.min(np.where(dates==dateBins[i])))
        dBmax.append(np.max(np.where(dates==dateBins[i])))

    timeStr = data[:,2]
    timeHour = []
    timeMin = []

    for j in np.arange(0,len(timeStr)):
        timeHour.append(int(timeStr[j][0:2]))
        timeMin.append(int(timeStr[j][3:5]))

    '''for each 0-23, from 0-29 and 30-59'''
    timeIndex = []

    for k in np.arange(0,len(timeHour)):
        if 0 <= timeMin[k] <= 29:
            timeIndex.append((timeHour[k]*2)+1)
        if 30 <=timeMin[k] <= 59:
            timeIndex.append((timeHour[k]*2)+2)

    '''with n days, assign one of n*48 possible bins to each data point'''
    idxCt = -1
    binIndex = []

    for l in np.arange(0,len(dateBins)):
        for m in np.arange(dBmin[l],dBmax[l]+1):
            idxCt = idxCt+1
            binIndex.append((l*48)+timeIndex[idxCt])

    '''now okay to deal with data'''
    elapsed = data[:,3]
    ctStr = data[:,4]
    counts = ctStr.astype(np.float)

    binCtCount = []    
    binCtMean = []
    for n in np.arange(0,maxBins):
        idx = [p for p, x in enumerate(binIndex) if x == n]
        binCtCount.append(len(idx))
        if n in binIndex:
            binCtMean.append(np.nanmean(counts[idx]))
        else:
            binCtMean.append(0)

    outputTXTFilename = f'{outDir}/{filename}_processed.txt'
    resultTXT = open(outputTXTFilename, 'w')

    outputCSVFilename = f'{outDir}/{filename}_processed.csv'
    resultCSV = open(outputCSVFilename, 'w')

    resultTXT.write("DayIndex\tTimeIndex\tAverageRead\n")
    resultCSV.write("DayIndex,TimeIndex,AverageRead\n")
    sampleIndex = -1;
    for sample in binCtMean:
        sampleIndex = sampleIndex+1
        resultTXT.write(f'{math.floor(sampleIndex/48)}\t{(sampleIndex%48/2.0):.2f}\t{sample}\n')
        resultCSV.write(f'{math.floor(sampleIndex/48)},{(sampleIndex%48/2.0):.2f},{sample}\n')
        
    print(f'Finished writing {outputTXTFilename}')
    print(f'Finished writing {outputCSVFilename}')

    '''Plot the data'''
    #plt.plot(binCtMean)
    scatterPlt = plt.subplot(311)
    plt.scatter(np.linspace(0, len(binCtMean) -1, num=len(binCtMean)), binCtMean, s=1.5)

    '''Sine equation against which to fit'''
    def sin_func(x, a, b, c, d):
        return a * np.sin(b * x + c) + d

    '''Fit whole data set'''
    startIndex = 0
    endIndex = len(binCtMean) - 1
    x_axis = np.linspace(startIndex, endIndex, num=len(binCtMean))
    midpoint = min(binCtMean) + (max(binCtMean) - min(binCtMean)) / 2.
    amplitudeGuess = max(binCtMean) - midpoint
    yOffsetGuess = midpoint
    periodGuess = math.pi * 2 / 48

    global_params, global_params_covariance = optimize.curve_fit(sin_func, x_axis, binCtMean, p0=[amplitudeGuess, periodGuess, 0, yOffsetGuess])
    error = np.sqrt(np.diag(global_params_covariance))
    errorString = [f'{err:3.2}' for err in error] 
    print(f'Estimated error rates of parameters {errorString}')
    plt.subplot(312, sharex=scatterPlt)
    plt.scatter(np.linspace(0, len(binCtMean) -1, num=len(binCtMean)), binCtMean, s=1.5, alpha=0.5)

    global_period = math.floor(math.pi * 2 / global_params[1])
    print(f'Global period for the data set is {global_period}')
    #global_period = 48

    def getXCoordsAtMinima(a, b, c, d, rangeStart, rangeEnd):
        period = abs(math.pi * 2 / b)
        mins = [optimize.fminbound(func = sin_func, x1=rangeStart, x2=rangeEnd, args = (a,b,c,d))]
        forward = mins[0]
        reverse = mins[0]
        while forward < rangeEnd:
            mins.append(forward)
            forward += period
        while reverse > rangeStart:
            mins.append(reverse)
            reverse -= period
        
        return sorted(list(set(mins)))

    def getXCoordsAtMaxima(a, b, c, d, rangeStart, rangeEnd):
        return getXCoordsAtMinima(-a, b, c, d, rangeStart, rangeEnd)

    p = plt.plot(x_axis, sin_func(x_axis, global_params[0], global_params[1], global_params[2], global_params[3]))

    minX = getXCoordsAtMinima(global_params[0], global_params[1], global_params[2], global_params[3], 0, len(binCtMean))
    minY = [sin_func(x, global_params[0], global_params[1], global_params[2], global_params[3]) for x in minX]
    plt.scatter(minX, minY, s=15, marker='v', c=p[0].get_color(), zorder=10) #The minimum

    maxX = getXCoordsAtMaxima(global_params[0], global_params[1], global_params[2], global_params[3], 0, len(binCtMean))
    maxY = [sin_func(x, global_params[0], global_params[1], global_params[2], global_params[3]) for x in maxX]
    plt.scatter(maxX, maxY, s=15, marker='^', c=p[0].get_color(), zorder=10) #The maximum
    
    plt.subplot(313, sharex=scatterPlt)
    plt.scatter(np.linspace(0, len(binCtMean) -1, num=len(binCtMean)), binCtMean, s=1.5, alpha=0.5)

    '''Output curve parameters'''
    outputCurvesFilename = f'{outDir}/{filename}_curves.csv'
    outputCurves = open(outputCurvesFilename, 'w')
    outputCurves.write('#the function params correspond to y = a*sin(b*x + c) + d\n')
    outputCurves.write('subjectiveDayOrGlobal, peak_or_trough, x, y, period, function_paramA, function_paramB, function_paramC, function_paramD\n')
        
    for i,_ in enumerate(minX):
        outputCurves.write(f'global, trough, {minX[i]}, {minY[i]}, {global_period}, {global_params[0]}, {global_params[1]}, {global_params[2]}, {global_params[3]}\n')
    for i,_ in enumerate(maxX):
        outputCurves.write(f'global, peak, {maxX[i]}, {maxY[i]}, {global_period}, {global_params[0]}, {global_params[1]}, {global_params[2]}, {global_params[3]}\n')

    '''Fit per day using global curve parameters'''
    for day in range(0, math.ceil(len(binCtMean) / global_period)):
        startIndex = day * global_period
        endIndex = (day + 1) * global_period - 1

        '''Extending the range a bit on either side'''
        startIndex = max(startIndex - math.floor(global_period/2), 0)
        endIndex = min(endIndex + math.floor(global_period/2), len(binCtMean))
        
        subset = binCtMean[startIndex:(endIndex+1)]

        x_axis = np.linspace(startIndex, endIndex, num=(endIndex - startIndex + 1))
    
        midpoint = min(subset) + (max(subset) - min(subset)) / 2.
        amplitudeGuess = max(subset) - midpoint
        yOffsetGuess = global_params[3]
        periodGuess = global_params[1]
    
        try:
            params, params_covariance = optimize.curve_fit(sin_func, x_axis, subset, p0=[amplitudeGuess, periodGuess, 0, yOffsetGuess])
            error = np.sqrt(np.diag(params_covariance))
            errorString = [f'{err:3.2}' for err in error] 
            print(f'Estimated error rates of parameters {errorString}')
            p = plt.plot(x_axis, sin_func(x_axis, params[0], params[1], params[2], params[3]))
            a = params[0]
            b = params[1]
            c = params[2]
            d = params[3]
            period = math.floor(math.pi * 2 / b)

            minX = getXCoordsAtMinima(a, b, c, d, startIndex, endIndex)
            minY = [sin_func(x, a, b, c, d) for x in minX]
            plt.scatter(minX, minY, s=15, marker='v', c=p[0].get_color(), zorder=10) #The minimum

            maxX = getXCoordsAtMaxima(a, b, c, d, startIndex, endIndex)
            maxY = [sin_func(x, a, b, c, d) for x in maxX]
            plt.scatter(maxX, maxY, s=15, marker='^', c=p[0].get_color(), zorder=10) #The maximum

            for i,_ in enumerate(minX):
                outputCurves.write(f'{day}, trough, {minX[i]}, {minY[i]}, {period}, {a}, {b}, {c}, {d}\n')
            for i,_ in enumerate(maxX):
                outputCurves.write(f'{day}, peak, {maxX[i]}, {maxY[i]}, {period}, {a}, {b}, {c}, {d}\n')
            
        except:
            print(f'Failed fitting curves to day {day} of {filename}')
        
    plotFilename = f'{outDir}/{filename}_plot.png'
    plt.savefig(plotFilename)
    plt.close()
    print(f'Finished plotting {plotFilename}')

'''argv contains a list of the arguments including the script name, so for example: "python script.py 1 2 3" would have argv = [script.py, 1, 2, 3]'''
if len(sys.argv) != 2:
    print("You need to specify exactly 1 argument:\n  python BinningTest.py file_or_folder_name")
    sys.exit()

Path = sys.argv[1]

filename, extension = os.path.splitext(Path)
if not extension:
    print(f'Extension-less input ({filename}) assumed to be folder')
    filesInFolder = [f for f in os.listdir(Path) if os.path.isfile(os.path.join(filename, f))]
    print(f'Found {len(filesInFolder)} files: {filesInFolder}')
    for file in filesInFolder:
        analyzeFile(f'{Path}/{file}')
else:
    '''If we're here it's a single file'''
    analyzeFile(os.path.abspath(Path))
