#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, copy, subprocess
from glob import glob
from datetime import datetime
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker

import astropy.wcs as wcs
from astropy.coordinates import SkyCoord

# parameters for testing in ipython. These are overwritten if run from
# the command line with proper arguments (see below)
# options are Sector Diff(1 or 0), or else Sector Cam CCD Diff(1 or 0)
sector = 27
cam = 0
ccd = 0
diffs = 0
#obj = '2MASX J07001137-6602251'
obj = None
rad = 10

# whether to make all images for the sector. If False, will stop after 1
# sample image for debugging and testing
makemovie = False
# put these images and movies in a separate, labeled 'test' directory
test = False

# which font properties to use
fontcol = 'white'
fontfile = 'Avenir-Black.otf'
    
# credits to put in the lower left corner
credit = 'By Ethan Kruse\n@ethan_kruse'

# resolution of the output videos
reso = 1080

# font sizes to use at various resolutions
# 1280x720, 1920x1080

# size of the neighboring chips labels in single chip mode, the left/right
# labels in full FOV mode, and the color bar label
fszs1 = {1080: 28}
# size of the movie title
fszs2 = {1080: 54}
# size of the date counter
fszs3 = {1080: 38}
# size of the CamX/CCDY labels in full FOV, the credits in lower left,
# and the numbers on the color bar
fszs4 = {1080: 21}
# the size of the 'data downlink gap' text
fszs5 = {1080: 84}
# size of the sector/camera/ccd info subtitles
fszs6 = {1080: 45}

# background color
bkcol = 'black'

# where is the data stored, and where should the output directories go?
if os.path.expanduser('~') == '/Users/ekruse':
    cd = '/Users/ekruse/Research/tessmovie'
    dataloc = '/Users/ekruse/Research/tessmovie/ffis/sector{0}'
    outdir = '/Users/ekruse/Research/tessmovie/movies/sector{0}/{1}'
elif os.path.expanduser('~') == '/Home/eud/ekruse1':
    cd = '/data/tessraid/ekruse1/tessmovies'
    dataloc = '/data/tessraid/data/ffis/sector{0}'
    outdir = '/data/tessraid/ekruse1/tessmovies/movies/sector{0}/{1}'
elif os.path.expanduser('~') == '/Users/ekruse1':
    cd = '/Users/ekruse1/tesseract/tessraid/ekruse1/tessmovies'
    dataloc = '/Users/ekruse1/tesseract/tessraid/data/ffis/sector{0}'
    outdir = '/Users/ekruse1/tesseract/tessraid/ekruse1/tessmovies/movies/sector{0}/{1}'
else:
    raise Exception('Need to set data and output paths for this computer.')
moviescript = os.path.join(cd, 'make_movie.sh')

# interpret the command line arguments to figure out what to run
if len(sys.argv) > 1:
    makemovie = True
    sector = int(sys.argv[1])
    assert len(sys.argv) == 5 or len(sys.argv) == 7
    cam = int(sys.argv[2])
    ccd = int(sys.argv[3])
    diffs = int(sys.argv[4])
    if len(sys.argv) > 5:
        obj = sys.argv[5]
        rad = int(sys.argv[6])
    else:
        obj = None
        rad = None

# in the full FOV plots, what text is on the left and right ends
# ie where is camera 1 and camera 4 pointing
leftlabel = 'ECLIPTIC'
if sector <= 13 or (sector >=27 and sector <= 39):
    rightlabel = 'SOUTH POLE'
elif sector >= 42 and sector <= 46:
    rightlabel = 'EAST'
    leftlabel = 'WEST'
else:
    rightlabel = 'NORTH POLE'
    
# output movie frame per second
if sector < 27:
    fps = 20
else:
    fps = 60

# color maps used for regular and diff movies. The diff map is modified
# to have black in the center later on
if obj is None:
    cmapstr = 'viridis'
else:
    cmapstr = 'gray'
diffcmapstr = 'coolwarm'

if obj is not None:
    skyco = SkyCoord.from_name(obj)
    coords = np.array([[skyco.ra.to_value(), skyco.dec.to_value()]])
else:
    coords = None

# data gap texts in order for each sector.
# S3 the first 4 days had some gaps while they experimented with
# pointing variations.
# S4 has a one cadence gap when they replaced the guide 
# star table.
dltxt = 'Data\nDownlink\nGap'
pstxt = 'Pointing\nStability\nExperiments'

gaptexts = {1: [dltxt], 2: [dltxt], 
            3: [pstxt, pstxt, pstxt, pstxt, dltxt, pstxt, pstxt],
            4: ['Guide Star\nTable\nReplaced',
                'Instrument\nAnomaly', dltxt],
            5: [dltxt], 6: [dltxt], 7: [dltxt],
            8: [dltxt, 'Instrument\nAnomaly'], 9: [dltxt],
            10: [dltxt], 11: [dltxt], 12:[dltxt], 13:[dltxt],
            14: [dltxt], 15: [dltxt], 16: [dltxt], 
            17: [dltxt, ''],
            18: ["TESS in\nEarth's\nShadow", dltxt],
            19: [dltxt, ''], 20: [dltxt, ''], 21: [dltxt, ''],
            22: ['', '', dltxt, ''], 23: [dltxt], 24: [dltxt],
            25: [dltxt], 26: [dltxt], 27: [dltxt], 28: [dltxt],
            29: [dltxt], 30: [dltxt], 31: [dltxt], 32: [dltxt],
            33: [dltxt], 34: [dltxt], 
            35: ["TESS Passes\nThrough\nEarth's Shadow", dltxt],
            36: [dltxt], 37: [dltxt], 38: [dltxt], 39: [dltxt],
            40: [dltxt], 41: [dltxt], 42: [dltxt], 43: [dltxt],
            44: [dltxt, ''], 45: [dltxt], 46: [dltxt],
            47: [dltxt], 48: [dltxt], 49: [dltxt], 
            50: ['Apogee\nDownlink\nTest', dltxt], 51: [dltxt],
            52: ['', dltxt], 53: [dltxt], 54: [dltxt],
            55: [dltxt]}

# minimum and maximum flux for the regular movies
vmin = 70
vmax = 1001.
# minimum and maximum flux for the difference image movies
if diffs:
    vmin = -100
    vmax = 100


# =======================
# end of input parameters
# =======================

# make sure the camera and CCD combination is valid for TESS
if sector < 1:
    print ('Bad Sector')
    sys.exit(1)
if cam not in np.arange(5) or ccd not in np.arange(5):
    print('Bad Cam or CCD')
    sys.exit(1)
# either both cam and ccd are 0 or they're both in the range of 1-4
if cam in np.arange(4) + 1 and ccd not in np.arange(4) + 1:
    print('Bad CCD')
    sys.exit(1)
if cam not in np.arange(4) + 1 and ccd in np.arange(4) + 1:
    print ('Bad Cam')
    sys.exit(1)

print(f'Sector {sector}, Cam {cam}, CCD {ccd}, Diff {diffs}')

# which version of the movie we're making here
if cam == 0 and ccd == 0:
    odir = 'all'
else:
    odir = 'cam{0}ccd{1}'.format(cam, ccd)

moviefile = f'sec{sector}' + odir

if diffs:
    odir += '_diff'
    moviefile += '_diff'

# allow for a testing directory
if test:
    odir += '_test'
    moviefile += '_test'
    
moviefile += '.mp4'

# set up the data locations and output directory
dataloc = dataloc.format(sector)
if obj is not None:
    outdir = outdir.format(sector, obj)
else:
    outdir = outdir.format(sector, odir)
    
# create the font we're using
fontfile = os.path.join(cd, fontfile)
prop = fm.FontProperties(fname=fontfile)

# make sure the directory is ready
if not os.path.exists(outdir):
    os.makedirs(outdir)

# clear out old images first
if makemovie:
    existing = glob(os.path.join(outdir, '*png'))
    for iexist in existing:
        os.remove(iexist)

# get our data files
if cam == 0 and ccd == 0:
    cstr = '?-?'
    single = False
else:
    cstr = '{0}-{1}'.format(cam, ccd)
    single = True
paths = os.path.join(dataloc, '*-s{0:04d}-{1}-*ffic.fits'.format(sector, cstr))
files = glob(paths)
files = np.array(files)

# XXX:
print('nfiles', len(files))

# what dates do they correspond to
dates = []
for ifile in files:
    dates.append(os.path.split(ifile)[1][4:].split('-')[0])

dates = np.array(dates)
udates = np.unique(dates)

# datetime versions of the dates
dtdates = []
for idate in udates:
    dtdates.append(datetime.strptime(idate, '%Y%j%H%M%S'))
delt = np.median(np.diff(dtdates))

# fill in any data gaps so our dates are equally spaced
gapdates = []
gapcads = []
gapstarts = []
gapends = []
gct = 0

for ii, idate in enumerate(dtdates[:-1]):
    gapstart = idate + delt
    gapend = None
    icad = ii*1
    while idate + 1.1*delt < dtdates[ii+1]:
        gapdates.append(datetime.strftime(idate+delt, '%Y%j%H%M%S'))
        gapcads.append(icad+1)
        idate += delt
        icad += 1
        gapend = idate
        
    if gapend is not None:
        print('Data gap detected: ', gapstart, 'to', gapend)
        print(f'Will have text:\n{gaptexts[sector][gct]}')
        gapstarts.append(gapstart - delt/2)
        # +0.6 for the middle of exposure that we use in the images to be less.
        # +1 to allow for cadence after gap in difference images is done
        # below in the actual comparison
        gapends.append(gapend + delt*0.6)
        gct += 1

assert gct == len(gaptexts[sector])

# final list of dates for the movie, including the data gaps
udates = np.unique(np.concatenate((udates, gapdates)))

""" 
Here's how the TESS cameras are laid out, with each chip/CCD labeled in its
proper location. I have rows and columns numbered in the grid on the top/right.

        0 1 

Cam 1:  4 3   0
        1 2   1
Cam 2:  4 3   2
        1 2   3
Cam 3:  2 1   4
        3 4   5
Cam 4:  2 1   6
        3 4   7
"""

# where each chip is located in x and y coordinates. 
# location is at index (camera # - 1)*4 + (CCD # - 1)
xlocs = np.array([1, 1, 0, 0, 3, 3, 2, 2, 4, 4, 5, 5, 6, 6, 7, 7])
ylocs = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1])
# whether the CCD image needs to be flipped/mirrored to get
# the correct orientation. No idea why this is true.
flips = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1])

# in full FOV plots, this is half the spacing between chips in the plots.
# in other words each chip takes up a 1 unit square, but the actual image
# has a side length of (1 - 2*gap) because of a gap on each side
gap = 0.02

# single chip vertical axis padding
# ie what percentage of the image height before the CCD plot starts and
# what percentage of the height from the top that the plot ends.
pad = 0.04

# where the single chip plot starts in percentage of image width
slx = 1. - 9/16*(1-pad)

# line width for the borders of the chips
if single:
    lw = 8
else:
    lw = 1.5

# how big the figure needs to be for the proper resolution output image
figsizes = {720: (8.54, 4.8), 1080: (19.2, 10.8)}

# set up the title text
if diffs:
    if single:
        titlestr = 'TESS: The Difference\nImage Movie'
    else:
        titlestr = 'TESS: The Difference Image Movie'
else:
    titlestr = 'TESS: The Movie'
if cam != 0 and ccd != 0:
    sectorstr = 'Sector {0}\nCamera {1}\nCCD {2}'.format(sector, cam, ccd)
else:
    sectorstr = 'Sector {0}'.format(sector)

# absoluate value of fluxes that are blacked out in the difference color bar.
cutout = 20

# create the difference image color bar with a blackout in the center
if diffs:
    diffcmap = copy.deepcopy(plt.get_cmap(diffcmapstr))
    
    # fractional region we need to cut out
    lcut = 0.5 - (-cutout/vmin)/2
    rcut = 0.5 + (cutout/vmax)/2
    
    cmcolors = ['red', 'green', 'blue']
    for icol in cmcolors:
        pops = []
        # chop out the regions of the color map that describe what to do
        # between lcut and rcut
        for ii in np.arange(len(diffcmap._segmentdata[icol])-1, -1, -1):
            if (diffcmap._segmentdata[icol][ii][0] > lcut and 
                    diffcmap._segmentdata[icol][ii][0] < rcut):
                pops.append(diffcmap._segmentdata[icol].pop(ii))
        # figure out where in the list to input our blackout
        for ii in np.arange(len(diffcmap._segmentdata[icol])):
            if diffcmap._segmentdata[icol][ii][0] > rcut:
                break
        # interpolate to figure out what the color is at our exact cut points
        rin = np.interp(rcut, [pops[0][0], diffcmap._segmentdata[icol][ii][0]],
                        [pops[0][2], diffcmap._segmentdata[icol][ii][1]])
        lin = np.interp(lcut, [diffcmap._segmentdata[icol][ii-1][0],
                               pops[-1][0]],
                        [diffcmap._segmentdata[icol][ii-1][2], pops[-1][1]])
        
        # insert our blackout commands into the color map
        diffcmap._segmentdata[icol].insert(ii, (rcut, 0, rin))
        diffcmap._segmentdata[icol].insert(ii, (lcut, lin, 0))
    
    # create our map
    diffcmap = colors.LinearSegmentedColormap('mymap', diffcmap._segmentdata)

# for storing old images
if diffs:
    olddata = [None]*len(xlocs)

# total number of difference images we're supposed to have if there's no
# data gap
if single:
    nodiff = 1
else:
    nodiff = len(xlocs)

plt.close('all')

startcad = None
startbjd = None
# go through every date and create the images
for ct, idate in enumerate(udates):
    # if we're debugging in ipython, break things off
    if diffs:
        if not makemovie and ct > 1:
            break
    else: 
        if not makemovie and ct > 0:
            break
    
    # grab the files for this date
    use = np.where(dates == idate)[0]
    use = use[np.argsort(files[use])]
    
    fig = plt.figure(1, figsize=figsizes[reso])
    # put the axis in the correct spot with the right background colors
    if single:
        ax = fig.add_axes([slx, 0+pad, 9/16*(1-2*pad), 1-2*pad])
    else:
        ax = fig.add_axes([0.005, 0, 0.99, 1])
    ax.axis('off')
    fig.patch.set_facecolor(bkcol)
    ax.patch.set_facecolor(bkcol)
    
    # how many images don't have a previous image for difference images
    nodiffct = 0
    
    # plot each CCD we want
    for ii, iuse in enumerate(use):
        with fits.open(files[iuse]) as ff:
            # index into the xlocs/ylocs/flips
            ind = (4 * (int(ff[1].header['camera'])-1) + 
                   int(ff[1].header['ccd']) - 1)
            # times of the image
            tstart = datetime.strptime(ff[0].header['date-obs'].split('.')[0],
                                        '%Y-%m-%dT%H:%M:%S')
            tend = datetime.strptime(ff[0].header['date-end'].split('.')[0],
                                        '%Y-%m-%dT%H:%M:%S')
            tmid = tstart + (tend - tstart)/2
            
            cadence = None
            tbjd = None
            try:
                cadence = ff[0].header['ffiindex']
                tbjd = ff[0].header['tstart'] + (ff[0].header['tstop'] - ff[0].header['tstart'])/2
            except KeyError:
                cadence = None
                tbjd = None
                
            data = ff[1].data * 1
            # clip out negative fluxes so the log color bar works
            if not diffs:
                data[data < 0.9*vmin] = 0.9*vmin
            
            if ct == 0:
                startcad = cadence
                startbjd = tbjd
                if coords is not None:
                    dwcs = wcs.WCS(ff[1])
                    loc = dwcs.all_world2pix(coords, 0)
                    loc = np.round(loc).astype(int)
                    loc = loc[0]
                    # because FITS and numpy flip axes
                    loc = loc[::-1]
                    
                    if (loc[0] < 0 or loc[0] >= data.shape[0] or loc[1] < 0 or
                            loc[1] >= data.shape[1]):
                        raise Exception(f'Object {obj} not in Cam {cam} CCD {ccd}')
                    xmin = max(0, loc[0] - rad)
                    xmax = min(data.shape[0], loc[0] + rad + 1)
                    ymin = max(0, loc[1] - rad)
                    ymax = min(data.shape[1], loc[1] + rad + 1)
            
            if obj is not None:
                data = data[xmin:xmax, ymin:ymax]
                    
    
            # the bounds of this chip
            if flips[ind]:
                extent = (xlocs[ind]+1-gap, xlocs[ind]+gap, 
                          ylocs[ind]+1-gap, ylocs[ind]+gap)
            else:
                extent = (xlocs[ind]+gap, xlocs[ind]+1-gap, 
                          ylocs[ind]+gap, ylocs[ind]+1-gap)
            
            # plot the difference image if possible
            if diffs:
                if olddata[ind] is not None:
                    cbpl = plt.imshow(data.T - olddata[ind].T, vmin=vmin,
                                      vmax=vmax, extent=extent, cmap=diffcmap)
                else:
                    # this chip couldn't be plotted
                    nodiffct += 1
                    # don't plot all the lines in the full FOV movies in the
                    # frame after a data gap
                    if not single:
                        olddata[ind] = data
                        continue
            # plot the chip
            else:
                cnorm = colors.LogNorm(vmin=vmin, vmax=vmax)
                cbpl = plt.imshow(data.T, norm=cnorm, #vmin=vmin, vmax=vmax,
                                  extent=extent, cmap=cmapstr)
            
            # plot the borders of this chip
            plt.plot([extent[0], extent[0]], [extent[2], extent[3]],
                     color='white', lw=lw)
            plt.plot([extent[1], extent[1]], [extent[2], extent[3]],
                     color='white', lw=lw)
            plt.plot([extent[0], extent[1]], [extent[2], extent[2]],
                     color='white', lw=lw)
            plt.plot([extent[0], extent[1]], [extent[3], extent[3]],
                     color='white', lw=lw)
            
            # label the chips in the full FOV plots
            txt = 'Cam {0}\nCCD {1}'.format(ff[1].header['camera'],
                                            ff[1].header['ccd'])
            if ylocs[ind] == 1:
                ytxt = 1.975
                va = 'bottom'
            else:
                ytxt = -0.005
                va = 'top'
                
            if not single:
                plt.text(xlocs[ind] + 0.5, ytxt, txt, ha='center', va=va,
                         color=fontcol, fontproperties=prop,
                         fontsize=fszs4[reso])
            # save this chip's data if we're doing difference images
            if diffs:
                olddata[ind] = data
    
    # tell the next iteration we had no previous images here
    if len(use) == 0 and diffs:
        olddata = [None]*len(xlocs)

    if (ct % 20) == 0:
        now = datetime.now().strftime("%d.%m.%Y %H:%M:%S") 
        print(f'{now}: Image {ct+1} of {len(udates)}. Sector {sector}, Cam {cam}, CCD {ccd}, Diff {diffs}')
    
    # skip the first image of the difference movies now that we've saved its
    # data for the next stage
    if diffs and ct == 0:
        plt.close(fig)
        continue
    
    # if we couldn't plot the (difference) image
    if len(use) == 0 or nodiffct == nodiff:
        # dates come from the file names which are the start of the exposure.
        # we want the mid time for plotting
        tmid = datetime.strptime(idate, '%Y%j%H%M%S') + delt/2
        # get the cadence number in the gap right
        if startcad is not None:
            cadence = startcad + gapcads[gapdates.index(idate)]
            tbjd = startbjd + gapcads[gapdates.index(idate)] * delt.total_seconds() / (24*60*60)

        if diffs:
            gadd = delt
        else:
            gadd = 0 * delt
            
        iind = -1
        gtxt = None
        # figure out what gap text to add to this
        for gg in np.arange(len(gapstarts)):
            if tmid > gapstarts[gg] and tmid < gapends[gg] + gadd:
                iind = gg
                gtxt = gaptexts[sector][gg]
                break
        if iind == -1 or gtxt is None:
            raise Exception(f'Time {tmid} does not fall in a known data gap.')
            
        # add the "data gap" text
        if single:
            # continue plotting the chip borders in single chip mode
            plt.plot([extent[0], extent[0]], [extent[2], extent[3]],
                     color='white', lw=lw)
            plt.plot([extent[1], extent[1]], [extent[2], extent[3]],
                     color='white', lw=lw)
            plt.plot([extent[0], extent[1]], [extent[2], extent[2]],
                     color='white', lw=lw)
            plt.plot([extent[0], extent[1]], [extent[3], extent[3]],
                     color='white', lw=lw)
            plt.text(slx + 9/32*(1-2*pad), 0.5, gtxt, 
                     transform=fig.transFigure, ha='center', va='center',
                     color=fontcol, fontproperties=prop, fontsize=fszs5[reso])
        else:
            plt.text(0.5, 0.5, gtxt, ha='center', va='center',
                     transform=fig.transFigure, color=fontcol,
                     fontproperties=prop, fontsize=fszs5[reso])

    # set proper limits on the plot
    if not single:
        plt.xlim(-0.14, 8.14)
        plt.ylim(-0.05, 2.05)
    else:
        if flips[ind]:
            plt.xlim(extent[1], extent[0])
            plt.ylim(extent[3], extent[2])
        else:
            plt.xlim(extent[0], extent[1])
            plt.ylim(extent[2], extent[3])
    
    # add in the title text, sector subtitle, and dates
    if not single:
        plt.text(0.5, 0.995, titlestr, transform=fig.transFigure, ha='center',
                 va='top', color=fontcol, fontproperties=prop, 
                 fontsize=fszs2[reso])
        plt.text(0.5, 0.92, sectorstr, transform=fig.transFigure, ha='center',
                 va='top', color=fontcol, fontproperties=prop, 
                 fontsize=fszs6[reso])
        tstr = tmid.strftime('%d %b %Y %H:%M')
        plt.text(0.5, 0.855, tstr, 
                 transform=fig.transFigure, ha='center', va='top', color=fontcol,
                 fontproperties=prop, fontsize=fszs3[reso])
        if cadence is not None:
            plt.text(0.992, 0.005, 'TBJD {0:.3f}\nCadence {1:d}'.format(tbjd, cadence),
                     transform=fig.transFigure, ha='right', va='bottom', color=fontcol,
                 fontproperties=prop, fontsize=fszs3[reso])
    else:
        # percentage down the plot to put the title text
        yshift = 0.05
        plt.text(slx/2, 0.99-yshift, titlestr, transform=fig.transFigure,
                 ha='center', va='top', color=fontcol, fontproperties=prop,
                 fontsize=fszs2[reso])
        tstr = tmid.strftime('%d %b %Y %H:%M')
        if cadence is not None:
            tstr += '\nCadence {0:d}'.format(cadence)
        if diffs:
            plt.text(slx/2, 0.80-yshift, sectorstr, transform=fig.transFigure,
                 ha='center', va='top', color=fontcol, fontproperties=prop,
                 fontsize=fszs6[reso])
            plt.text(slx/2, 0.58-yshift, 
                     tstr, color=fontcol,
                     transform=fig.transFigure, fontproperties=prop,
                     ha='center', va='top', fontsize=fszs3[reso])
        else:
            plt.text(slx/2, 0.877-yshift, sectorstr, transform=fig.transFigure,
                 ha='center', va='top', color=fontcol, fontproperties=prop,
                 fontsize=fszs6[reso])
            plt.text(slx/2, 0.657-yshift, tstr,
                     transform=fig.transFigure, ha='center', va='top', 
                     color=fontcol, fontproperties=prop, fontsize=fszs3[reso])
    
    # label the neighboring chips in single CCD mode
    if single:
        # properties for the pointing arrows
        arrow = dict(arrowstyle="wedge, tail_width=2, shrink_factor=0.5",
                     color=fontcol)
        # because the bottom text doesn't seem centered
        tyoff = -pad/12
        # top neighbor
        if ylocs[ind] == 0:
            neighb = np.where((xlocs == xlocs[ind]) & (ylocs == 1))[0][0]
            ncam = neighb // 4 + 1
            nccd = (neighb % 4) + 1
            ntxt = 'Cam {0}     CCD {1}'.format(ncam, nccd)
            plt.text(slx + 9/32*(1-2*pad), 1-pad/2+tyoff, ntxt,
                     transform=fig.transFigure, ha='center', va='center',
                     color=fontcol, fontproperties=prop, fontsize=fszs1[reso])
            ax.annotate("", xy=(slx + 9/32*(1-2*pad), 1-pad/4+tyoff), 
                        xycoords='figure fraction', ha='center', va='center',
                        xytext=(slx + 9/32*(1-2*pad), 1-pad*3/4+tyoff),
                        textcoords='figure fraction', arrowprops=arrow, 
                        annotation_clip=False, color=fontcol)
        else:
            # bottom neighbor
            neighb = np.where((xlocs == xlocs[ind]) & (ylocs == 0))[0][0]
            ncam = neighb // 4 + 1
            nccd = (neighb % 4) + 1
            ntxt = 'Cam {0}     CCD {1}'.format(ncam, nccd)
            plt.text(slx + 9/32*(1-2*pad), pad/2+tyoff, ntxt, 
                     transform=fig.transFigure, ha='center', va='center',
                     color=fontcol, fontproperties=prop, fontsize=fszs1[reso])
            ax.annotate("", xy=(slx + 9/32*(1-2*pad), pad/4+tyoff),
                        xycoords='figure fraction', ha='center', va='center',
                        xytext=(slx + 9/32*(1-2*pad), pad*3/4+tyoff),
                        textcoords='figure fraction', arrowprops=arrow, 
                        annotation_clip=False, color=fontcol)
        # right neighbor
        neighb = np.where((xlocs == xlocs[ind] + 1) & (ylocs == ylocs[ind]))[0]
        if len(neighb) > 0:
            if len(neighb) > 1:
                raise Exception('Multiple right neighboring chips?')
            neighb = neighb[0]
            ncam = neighb // 4 + 1
            nccd = (neighb % 4) + 1
            ntxt = 'Cam{0}\nCCD{1}'.format(ncam, nccd)
            ntxtstr = ''
            for char in ntxt:
                ntxtstr += char + '\n'
            ntxtstr = ntxtstr[:-1]
            plt.text(slx + 9/16*(1-2*pad)+pad/2*9/16, 0.5, ntxtstr, 
                     transform=fig.transFigure, ha='center', va='center',
                     color=fontcol, fontproperties=prop, fontsize=fszs1[reso],
                     multialignment='center')
            ax.annotate("", xy=(slx + 9/16*(1-2*pad)+pad*3/4*9/16, 0.5), 
                        xycoords='figure fraction', ha='center', va='center',
                        xytext=(slx + 9/16*(1-2*pad)+pad/4*9/16, 0.5), 
                        textcoords='figure fraction', arrowprops=arrow, 
                        annotation_clip=False, color=fontcol)
        # left neighbor
        neighb = np.where((xlocs == xlocs[ind] - 1) & (ylocs == ylocs[ind]))[0]
        if len(neighb) > 0:
            if len(neighb) > 1:
                raise Exception('Multiple left neighboring chips?')
            neighb = neighb[0]
            ncam = neighb // 4 + 1
            nccd = (neighb % 4) + 1
            ntxt = 'Cam{0}\nCCD{1}'.format(ncam, nccd)
            ntxtstr = ''
            for char in ntxt:
                ntxtstr += char + '\n'
            ntxtstr = ntxtstr[:-1]
            plt.text(slx - pad/2*9/16, 0.5, ntxtstr, transform=fig.transFigure,
                     ha='center', va='center', color=fontcol,
                     fontproperties=prop, fontsize=fszs1[reso],
                     multialignment='center')
            ax.annotate("", xy=(slx - pad*3/4*9/16, 0.5), 
                        xycoords='figure fraction', ha='center', va='center',
                        xytext=(slx - pad/4*9/16, 0.5), 
                        textcoords='figure fraction', arrowprops=arrow,
                        annotation_clip=False, color=fontcol)
    
    # in full FOV mode add the right and left labels if not in a data gap
    if not single and len(use) > 0 and nodiffct != nodiff:
        # too lazy to figure out how to do this right
        # put newlines between every letter in the left and right labels in 
        # full FOV mode
        lstr = ''
        for char in leftlabel:
            lstr += char + '\n'
        lstr = lstr[:-1]
        
        rstr = ''
        for char in rightlabel:
            rstr += char + '\n'
        rstr = rstr[:-1]
        
        plt.text(0.002, 0.5, lstr, transform=fig.transFigure, ha='left', 
                 va='center', multialignment='center', color=fontcol, 
                 fontproperties=prop, fontsize=fszs1[reso])
        plt.text(0.994, 0.5, rstr, transform=fig.transFigure, ha='right', 
                 va='center', multialignment='center', color=fontcol, 
                 fontproperties=prop, fontsize=fszs1[reso])
    
    # add the credits in the lower left
    plt.text(0.008, 0.01, credit, transform=fig.transFigure, ha='left', 
             va='bottom', multialignment='left', color=fontcol, 
             fontproperties=prop, fontsize=fszs4[reso])
    
    # put in the color bar axis
    if not single:
        ax2 = fig.add_axes([0.35, 0.1, 0.3, 0.05])
    else:
        ax2 = fig.add_axes([(slx-0.3)/2, 0.2, 0.3, 0.05])

    # set up the tick marks we want to label
    if diffs:
        ticks = np.array([vmin, -cutout, cutout, vmax]).astype(int)
        ticklabs = []
        for itick in ticks:
            ticklabs.append(str(itick))
    else:
        ticks = np.arange((vmin//100)*100 + 100, vmax, 100).astype(int)
        ticklabs = []
        for itick in ticks:
            ticklabs.append(str(itick))
        for ii in np.arange(1, len(ticklabs)-1):
            ticklabs[ii] = ''
    
    # put in the color bar and adjust everything
    cbar = plt.colorbar(cbpl, orientation='horizontal', cax=ax2, extend='both')
    if diffs:
        cbar.set_label('Change in Flux', color=fontcol, fontproperties=prop,
                       fontsize=fszs1[reso])
    else:
        cbar.set_label('Calibrated Flux', color=fontcol, fontproperties=prop,
                       fontsize=fszs1[reso])
        
    cbar.outline.set_edgecolor(fontcol)
    cbar.outline.set_linewidth(3)
    cbar.set_ticks(ticker.LogLocator())
    # for some reason need to clear the original ticks because I can't figure 
    # out a way to change or remove them properly.
    cbar.ax.xaxis.set_ticks([])
    cbar.ax.xaxis.set_ticks([], minor=True)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabs)
    cbar.ax.tick_params(color=fontcol)
    cbar.ax.set_xticklabels(ticklabs, color=fontcol, fontproperties=prop,
                            fontsize=fszs4[reso], horizontalalignment='center')
    cbar.ax.tick_params(axis='x', color=fontcol, width=3, length=6, zorder=5)
    # why do these minor ticks keep coming back as little black overlays
    cbar.ax.xaxis.set_ticks([], minor=True)
    # for some reason need to set each label's parameters individually now?
    for icbl in cbar.ax.get_xticklabels(which='both'):
        icbl.set_fontproperties(prop)
        icbl.set_fontsize(fszs4[reso])

    cbar.ax.patch.set_alpha(0.)    
    # save the output image and then close it to save memory
    outstr = os.path.join(outdir, 'img{0:05d}.png'.format(ct))
    fig.savefig(outstr, facecolor=fig.get_facecolor(), edgecolor='none')
        
    if makemovie:
        plt.close(fig) 

# create the movie
if makemovie:
    import shlex
    if os.path.exists(os.path.join(outdir, moviefile)):
        os.remove(os.path.join(outdir, moviefile))
    command = moviescript + f' {shlex.quote(outdir+"/")} {moviefile} {fps}'
    subprocess.check_call(command, shell=True)

