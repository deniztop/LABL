import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import pathlib
import pywt
import gc
from scipy import stats
from scipy import optimize
import scaleogram as scg

def plotSignal(zeroPoint, hours, signal, title):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
    
    fig.subplots_adjust(hspace=0.4)
    time = np.array(hours)
    time = time - zeroPoint
    ax1.plot(time, signal)
    ax1.set_title(title)
    ax1.set_xticks(np.arange(0, time[-1], 24))
    ax1.set_xlabel(f"Time [hours] since {zeroPoint} hours")

    plt.sca(ax2) #making ax2 current to prevent calls to plt from leaking to other axes inside cws
    ax2 = scg.cws(time, signal, scales=scales, wavelet=wavelet,
            ax=ax2, cmap="jet", cbar=None, yaxis='period',
            title=f'CWT using {wavelet}', xlabel=f"Time [hours] since {zeroPoint} hours")

    '''For ridge analysis we do a cwt manually'''
    coefs, scales_freq = pywt.cwt(signal, scales, wavelet, sampling_period=(1/samplesPerHour))

    scales_period = 1./scales_freq
    values = np.abs(coefs)
    maxIndices = np.argmax(values, axis=0)
    maxValues = np.max(values, axis=0)

    ridge = np.array([scales_period[index] for index in maxIndices])

    maxValue = max(maxValues)
    minValue = min(maxValues)

    """
    COI implementation courtesy of:

    Copyright (c) 2019 Alexandre Sauve <http://github.com/alsauve/scaleogram>

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    # convert the wavelet scales frequency into time domain periodicity
    scales_coi = scales_period
    max_coi  = scales_coi[-1]
    dt = time[1]-time[0]

    # produce the line and the curve delimiting the COI masked area
    xmesh = np.concatenate([time, [time[-1]+dt]])
    mid = int(len(xmesh)/2)
    time0 = np.abs(xmesh[0:mid+1]-xmesh[0])
    ymask = np.zeros(len(xmesh), dtype=np.float16)
    ymhalf= ymask[0:mid+1]  # compute the left part of the mask
    ws    = np.argsort(scales_period) # ensure np.interp() works
    minscale, maxscale = sorted(ax2.get_ylim())
    ymhalf[:] = np.interp(time0,
            scales_period[ws], scales_coi[ws])
    yborder = np.zeros(len(xmesh)) + maxscale
    ymhalf[time0 > max_coi]   = maxscale
    
    # complete the right part of the mask by symmetry
    ymask[-mid:] = ymhalf[0:mid][::-1]

    # plot the mask and forward user parameters
    ax3.plot(xmesh, ymask)
    coikw = {
        'alpha':'0.5',
        'hatch':'/',
    }
    ax3.fill_between(xmesh, yborder, ymask, **coikw)

    '''
    End COI Implementation
    '''

    passing = []
    for i, r in enumerate(ridge):
        border = yborder[i]
        mask = ymask[i]
        if mask < r < border:
            continue
        else:
            passing.append(i)
    passing = np.array(passing)
    
    if passing.size > 0:
        ridge = ridge[passing]
        time = time[passing]
        maxValues = maxValues[passing]
        
        intercept, slope = np.polynomial.polynomial.polyfit(time, ridge, 1, w=maxValues)

        threshold = minValue + (maxValue - minValue) * percentCutoff #cut off the bottom portion based on %
        passing = np.where(maxValues > threshold)
        trimmedRidge = ridge[passing]
        trimmedTime = time[passing]
        trimmedValues = maxValues[passing]
        ax2.invert_yaxis()
        ax2.set_yticks(np.arange(smallestScaleInSampleUnits / samplesPerHour, largestScaleInSampleUnits / samplesPerHour, 1))
        ax2.set_ylabel("Period [hours]")
        ax2.plot(trimmedTime, trimmedRidge, color='white', alpha=0.7)
        ax3.scatter(trimmedTime, trimmedRidge, s=10, color='blue')

        minY = np.min(ridge)
        maxY = np.max(ridge)
        ax3.set_ylim(minY, maxY)
        ax3.plot(time, line_func(time, slope, intercept))
        ax3.set_title(f'Fitted Ridge Line: y = {slope:.6f}*x + {intercept:.6f}')
        ax3.set_yticks(np.arange(smallestScaleInSampleUnits / samplesPerHour, largestScaleInSampleUnits / samplesPerHour, 1))
        ax3.set_xlabel(f"Time [hours] since {zeroPoint} hours")
        ax3.set_ylabel("Period [hours]")

        #ensure that the two lower figures share y-axes
        ax2.get_shared_y_axes().join(ax2, ax3)

        for ax in (ax1, ax2, ax3):
            ax.set_xlim(0, math.ceil(hours[-1]))
    else:
        print(f'{title} failed, no data lies in valid region')

    return trimmedTime, trimmedRidge, trimmedValues

def line_func(x, a, b):
    return a*x + b 

def floatInputOrDefault(default_value):
    userIn = input()
    return float(userIn) if userIn else default_value

def signalFromFile(filepath):
    strippedPath = 'stripped.temp'
    with open(filepath) as infile, open(strippedPath, 'w') as outfile:
        for line in infile:
            if not line.strip(): continue #skip
            outfile.write(line)

    data = np.genfromtxt(strippedPath,dtype='str',delimiter=",", skip_header=2)
    os.remove(strippedPath)
    signal = [float(x) for x in data[:,4]]
    hours = [float(x) * 24.0 for x in data[:,3]]
    return hours, signal

WAVELET_STEP = 0
ANALYSIS_STEP = 1
TEST_STEP = 2
RUN_STEP = 3

'''Default configuration'''
defaultConfiguration = {"bandwidthParameter":.7, "centralFrequency":1, "samplesPerHour":(60/4), "zeroPointHours":10, "largestPeriodOfInterestInHours":32, "smallestPeriodOfInterestHours":16, "samplesPerScaleIncrement":1, "percentCutoff":.25}
configuration = defaultConfiguration
firstStep = WAVELET_STEP
if len(sys.argv) > 1:
    configFile = sys.argv[1]
    if not os.path.exists(configFile):
        print(f'User provided configuration file {configFile} did not exist.')
        sys.exit()

    configuration = eval(open(configFile, 'r').read())
    print(f'User provided configuration: {configuration}')
    firstStep = TEST_STEP

'''
The following parameters are as per:
https://github.com/alsauve/scaleogram/blob/master/doc/scale-to-frequency.ipynb
and https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#complex-morlet-wavelets
'''
bandwidthParameter = configuration["bandwidthParameter"]
centralFrequency = configuration["centralFrequency"]
wavelet = f'cmor{bandwidthParameter}-{centralFrequency}' # becomes "cmorB-C" e.g. cmor1-1
def waveletConfig():
    global bandwidthParameter
    global centralFrequency
    global wavelet
    print('1 of 4: Wavelet configuration step. We will configure the bandwidth parameter and central frequency for the mother wavelet. \n See: https://github.com/alsauve/scaleogram/blob/master/doc/scale-to-frequency.ipynb \n Or: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#complex-morlet-wavelets')
    print(f'bandwidth parameter [defaulted to {bandwidthParameter}]:')
    bandwidthParameter = floatInputOrDefault(bandwidthParameter) #aka B
    print(f'central frequency [defaulted to {centralFrequency}]:')
    centralFrequency = floatInputOrDefault(centralFrequency) #aka C
    wavelet = f'cmor{bandwidthParameter}-{centralFrequency}' # becomes "cmorB-C" e.g. cmor1-1
    print(f'Using wavelet: {wavelet}')
    return ANALYSIS_STEP

'''
The following parameters indicate the range of scales we're interested in. The 'centralFrequency' above
 represents the frequency at which the wavelet is most sensitive at scale 1.0. Subsequent scales will have
 the wavelet scaled up and will capture longer periods.
'''
samplesPerHour = configuration["samplesPerHour"]
zeroPointHours = configuration["zeroPointHours"]
largestPeriodOfInterestInHours = configuration["largestPeriodOfInterestInHours"]
smallestPeriodOfInterestHours = configuration["smallestPeriodOfInterestHours"]
largestScaleInSampleUnits = largestPeriodOfInterestInHours * samplesPerHour * centralFrequency
smallestScaleInSampleUnits = smallestPeriodOfInterestHours * samplesPerHour * centralFrequency
samplesPerScaleIncrement = configuration["samplesPerScaleIncrement"]
percentCutoff = configuration["percentCutoff"]
scales = np.arange(1, largestScaleInSampleUnits, samplesPerScaleIncrement)
def analysisConfig():
    global samplesPerHour
    global zeroPointHours
    global smallestPeriodOfInterestHours
    global largestPeriodOfInterestInHours
    global largestScaleInSampleUnits
    global samplesPerScaleIncrement
    global scales
    global percentCutoff
    print('2 of 4: Analysis configuration step. We will configure the parameters to be used in the CWT analysis. These will determine the wavelet scales and ridge line.')
    print(f'Data points in a given hour. For example, samples at 4 minute intervals would yield 15 samples in an hour. [defaulted to {samplesPerHour}]:')
    samplesPerHour = floatInputOrDefault(samplesPerHour) #60 minutes per hour and the data is at 4 minute intervals
    print(f'The point, in hours, at which to place the zero hour mark. [defaulted to {zeroPointHours}]:')
    zeroPointHours = floatInputOrDefault(zeroPointHours)
    print(f'Period of shortest wave to match against, in hours. This should be smaller than the smallest expected signal. [defaulted to {smallestPeriodOfInterestHours}]:')
    smallestPeriodOfInterestHours = floatInputOrDefault(smallestPeriodOfInterestHours)
    smallestScaleInSampleUnits = smallestPeriodOfInterestHours * samplesPerHour * centralFrequency
    print(f'Period of the longest wave to match against, in hours. This should exceed the largest expected signal. [defaulted to {largestPeriodOfInterestInHours}]:')
    largestPeriodOfInterestInHours = floatInputOrDefault(largestPeriodOfInterestInHours)
    largestScaleInSampleUnits = largestPeriodOfInterestInHours * samplesPerHour * centralFrequency
    print(f'Given those parameters the largest period in units of samples (recall: {samplesPerHour} in an hour) that we will analyse is {largestScaleInSampleUnits}.')
    print(f'Select the increment from {smallestScaleInSampleUnits} to {largestScaleInSampleUnits}. The smaller the increment, the higher the resolution but the slower the analysis. [defaulted to {samplesPerScaleIncrement}]:')
    samplesPerScaleIncrement = floatInputOrDefault(samplesPerScaleIncrement)
    scales = np.arange(smallestScaleInSampleUnits, largestScaleInSampleUnits, samplesPerScaleIncrement)
    print(f'Analysis will look at {len(scales)} total scales. From {scales[0]} to {scales[-1]}')
    print(f'Finally, below which percent threshold, expressed as a decimal [0..1], should the coefficients for the ridge line be discarded? [defaulted to {percentCutoff}]:')
    percentCutoff = floatInputOrDefault(percentCutoff)
    return TEST_STEP

def testStep():
    print('3 of 4: Test step. Would you like to run a test to validate your inputs? [y/n]:')
    desire = str(input())
    if (desire == 'n'):
        return RUN_STEP
    elif (desire == 'y'): 
        signal = []
        hours = []
        print('Full path to sample data file:')
        filename = str(input());
        hours, signal = signalFromFile(filename)
        plotSignal(zeroPointHours, hours, signal, filename)
        plt.show()
        plt.close()

        print('Were you happy with the result? A response of no will restart the configuration steps [y/n]:')
        desire = ''
        while not (desire == 'n' or desire == 'y'):
            desire = str(input())
            if (desire == 'n'):
                return WAVELET_STEP
            elif (desire == 'y'):
                return RUN_STEP
            else:
                print("Please input 'y' or 'n'")
    else:
        print("Please input 'y' or 'n'")
        return TEST_STEP

steps = [waveletConfig, analysisConfig, testStep]

currentStep = firstStep
while currentStep != RUN_STEP:
    gc.collect()
    currentStep = steps[currentStep]()

configuration["bandwidthParameter"] = bandwidthParameter
configuration["centralFrequency"] = centralFrequency
configuration["samplesPerHour"] = samplesPerHour
configuration["zeroPointHours"] = zeroPointHours
configuration["largestPeriodOfInterestInHours"] = largestPeriodOfInterestInHours
configuration["smallestPeriodOfInterestHours"] = smallestPeriodOfInterestHours
configuration["samplesPerScaleIncrement"] = samplesPerScaleIncrement
configuration["percentCutoff"] = percentCutoff
print(f'Would you like to save the current configuration: {configuration}? [y/n]')
desire = ''
while not (desire == 'n' or desire == 'y'):
    desire = str(input())
    if (desire == 'n'):
        print('Not saved.')
    elif (desire == 'y'):
        print('Enter a filename:')
        fname = f'{str(input())}.cfg'
        with open(fname, 'w') as f:f.write(repr(configuration))
        print(f'Wrote configuration to {fname}. You may pass it in to future runs')
    else:
        print("Please input 'y' or 'n'")

pwd = os.getcwd()
outdir = f'{pwd}/scaleograms'
pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) 
print(f'Results will be in {outdir}')

print('Running full analysis.')
print('Provide the path to the folder containing ONLY your data files:')
folderpath = ''
while not os.path.exists(folderpath):
    folderpath = str(input())
    if not os.path.exists(folderpath):
        print(f'Folder: ({folderpath}) does not appear to exist. Try again.')

files = [os.path.join(folderpath, f) for f in os.listdir(folderpath) if os.path.isfile(os.path.join(folderpath, f))]
print(f'Found {len(files)} files...')
for filepath in files:
    print(f'Processing {filepath}...')
    gc.collect()
    hours, signal = signalFromFile(filepath)
    filename = os.path.splitext(os.path.basename(filepath))[0]

    time, ridge, coef = plotSignal(zeroPointHours, hours, signal, filename)

    plt.savefig(f'{outdir}/{filename}_scaleogram.png')
    plt.close()
    gc.collect()

    zipped = np.array(list(zip(time, ridge, coef)));

    np.savetxt(f'{outdir}/{filename}_ridge.txt', zipped, delimiter=',')
print(f'Finished')