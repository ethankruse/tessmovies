#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:59:08 2018

@author: ekruse
"""


import matplotlib.colors as colors
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from datetime import datetime
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import sys
import copy

sector = 1
cam = 1
ccd = 1
makemovie = False
diffs = True

cmap = 'viridis'
#cmap = 'plasma'
#cmap = 'gray'
diffcmap = 'BrBG'
diffcmap = 'coolwarm'

fontcol = 'white'
fontfile = 'Avenir-Black.otf'

leftlabel = 'ECLIPTIC'
rightlabel = 'SOUTH POLE'

credit = 'By Ethan Kruse\n@ethan_kruse'

reso = 1080

# font sizes at various resolutions
# 1280x720, 1920x1080
fszs1 = {1080: 28}
fszs2 = {1080: 42}
fszs3 = {1080: 35}
fszs4 = {1080: 21}
# the size of the 'data downlink gap' text in single chip mode
fszs5 = {1080: 84}

# background color
bkcol = 'black'


if len(sys.argv) > 1:
    makemovie = True
    sector = int(sys.argv[1])
    assert len(sys.argv) == 3 or len(sys.argv) == 5
    if len(sys.argv) > 3:
        cam = int(sys.argv[2])
        ccd = int(sys.argv[3])
        diffs = int(sys.argv[4])
    else:
        cam = 0
        ccd = 0
        diffs = int(sys.argv[2])
        
if cam in np.arange(4) + 1 and ccd not in np.arange(4) + 1:
    print('Bad CCD')
    sys.exit(1)
if cam not in np.arange(4) + 1 and ccd in np.arange(4) + 1:
    print ('Bad Cam')
    sys.exit(1)
if sector < 1:
    print ('Bad Sector')
    sys.exit(1)
if cam not in np.arange(5) or ccd not in np.arange(5):
    print('Bad Cam or CCD')
    sys.exit(1)
if cam == 0 and ccd == 0:
    odir = 'all'
else:
    odir = 'Cam{0}CCD{1}'.format(cam, ccd)

if diffs:
    odir += 'Diffs'

if os.path.expanduser('~') == '/Users/ekruse':
    cd = '/Users/ekruse/Research/tessmovie'
    dataloc = '/Users/ekruse/Research/tessmovie/ffis/sector{0}'.format(sector)
    outdir = '/Users/ekruse/Research/tessmovie/movies/sector{0}/{1}'.format(sector, odir)
    
elif os.path.expanduser('~') == '/Home/eud/ekruse1':
    cd = '/data/tessraid/ekruse1/tessmovies'
    dataloc = '/data/tessraid/data/ffis/sector{0}'.format(sector)
    outdir = '/data/tessraid/ekruse1/tessmovies/movies/sector{0}/{1}'.format(sector, odir)

fontfile = os.path.join(cd, fontfile)

print(f'Sector {sector}, Cam {cam}, CCD {ccd}')



if not os.path.exists(outdir):
    os.mkdir(outdir)
    
if makemovie:
    existing = glob(os.path.join(outdir, '*png'))
    for iexist in existing:
        os.remove(iexist)

prop = fm.FontProperties(fname=fontfile)

plt.close('all')

if cam == 0 and ccd == 0:
    cstr = '?-?'
    single = False
else:
    cstr = '{0}-{1}'.format(cam, ccd)
    single = True
    
paths = os.path.join(dataloc, '*-s{0:04d}-{1}-*ffic.fits'.format(sector, cstr))
files = glob(paths)
files = np.array(files)
dates = []
for ifile in files:
    dates.append(os.path.split(ifile)[1][4:].split('-')[0])

dates = np.array(dates)

udates = np.unique(dates)

dtdates = []
for idate in udates:
    dtdates.append(datetime.strptime(idate, '%Y%j%H%M%S'))

delt = np.median(np.diff(dtdates))

gapdates = []
for ii, idate in enumerate(dtdates[:-1]):
    while idate + 1.5*delt < dtdates[ii+1]:
        gapdates.append(datetime.strftime(idate+delt, '%Y%j%H%M%S'))
        idate += delt

udates = np.unique(np.concatenate((udates, gapdates)))


"""
vmax = 1000.

idate = udates[24]

use = np.where(dates == idate)[0]
use = use[np.argsort(files[use])]

fig = plt.figure()

with fits.open(files[use[7]]) as ff:
    wcs = WCS(ff[1].header)
    ax = plt.subplot(projection=wcs)
    
for ii, iuse in enumerate(use):
    with fits.open(files[iuse]) as ff:
        print(ff[1].header['camera'], ff[1].header['ccd'], ff[1].header['crval1'], ff[1].header['crval2'])
        wcs = WCS(ff[1].header)
        data = ff[1].data * 1
        data[data < 1] = 1.

        plt.imshow(data, norm=colors.LogNorm(vmin=75, vmax=vmax), 
                   transform=ax.get_transform(wcs), cmap=cmap)
        txt = 'cam {0} ccd {1}'.format(ff[1].header['camera'], ff[1].header['ccd'])
        #plt.text(ff[1].header['crval1'], ff[1].header['crval2'], txt, transform=ax.get_transform(wcs))

ax.set_xlim(-3000, 5000)
ax.set_ylim(-6000, 10000)
plt.colorbar()
"""



""" 0 1 

1:  4 3  0
    1 2  1
2:  4 3  2
    1 2  3
3:  2 1  4
    3 4  5
4:  2 1  6
    3 4  7
"""

xlocs = np.array([1, 1, 0, 0, 3, 3, 2, 2, 4, 4, 5, 5, 6, 6, 7, 7])
ylocs = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1])
flips = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1])
gap = 0.02
# single chip axis padding
pad = 0.04
# padding on the right side of single chips for text
#srtpad = 0.065
srtpad = 0.
# where the single chip plot starts
#slx = 7/16+pad-srtpad
slx = 1. - 9/16*(1-2*pad) - srtpad - pad*9/16
vmin = 70
vmax = 1001.

if diffs:
    vmin = -100
    vmax = 100
cutout = 20

if diffs:
    diffcmap = copy.deepcopy(plt.get_cmap(diffcmap))
    lcut = 0.5 - (-cutout/vmin)/2.
    rcut = 0.5 + (cutout/vmax)/2
    
    for ii in np.arange(len(diffcmap._segmentdata['red'])-1, -1, -1):
        if diffcmap._segmentdata['red'][ii][0] > lcut and diffcmap._segmentdata['red'][ii][0] < rcut:
            diffcmap._segmentdata['red'].pop(ii)
    for ii in np.arange(len(diffcmap._segmentdata['blue'])-1, -1, -1):
        if diffcmap._segmentdata['blue'][ii][0] > lcut and diffcmap._segmentdata['blue'][ii][0] < rcut:
            diffcmap._segmentdata['blue'].pop(ii)
    for ii in np.arange(len(diffcmap._segmentdata['green'])-1, -1, -1):
        if diffcmap._segmentdata['green'][ii][0] > lcut and diffcmap._segmentdata['green'][ii][0] < rcut:
            diffcmap._segmentdata['green'].pop(ii)
    
    for ii in np.arange(len(diffcmap._segmentdata['red'])):
        if diffcmap._segmentdata['red'][ii][0] > rcut:
            break
    # technically to do this right, need to interpolate a bit
    diffcmap._segmentdata['red'].insert(ii, (rcut, 0, diffcmap._segmentdata['red'][ii][1]))
    diffcmap._segmentdata['red'].insert(ii, (lcut, diffcmap._segmentdata['red'][ii-1][2], 0))
    
    for ii in np.arange(len(diffcmap._segmentdata['blue'])):
        if diffcmap._segmentdata['blue'][ii][0] > rcut:
            break
    # technically to do this right, need to interpolate a bit
    diffcmap._segmentdata['blue'].insert(ii, (rcut, 0, diffcmap._segmentdata['blue'][ii][1]))
    diffcmap._segmentdata['blue'].insert(ii, (lcut, diffcmap._segmentdata['blue'][ii-1][2], 0))
    
    for ii in np.arange(len(diffcmap._segmentdata['green'])):
        if diffcmap._segmentdata['green'][ii][0] > rcut:
            break
    # technically to do this right, need to interpolate a bit
    diffcmap._segmentdata['green'].insert(ii, (rcut, 0, diffcmap._segmentdata['green'][ii][1]))
    diffcmap._segmentdata['green'].insert(ii, (lcut, diffcmap._segmentdata['green'][ii-1][2], 0))
    
    diffcmap = colors.LinearSegmentedColormap('tet', diffcmap._segmentdata)
    


if diffs:
    if single:
        titlestr = 'TESS: The Difference\nImage Movie\nSector {0}'.format(sector)
    else:
        titlestr = 'TESS: The Difference Image Movie\nSector {0}'.format(sector)
else:
    titlestr = 'TESS: The Movie\nSector {0}'.format(sector)
if cam != 0 and ccd != 0:
    titlestr += '\nCamera {0}\nCCD {1}'.format(cam, ccd)


olddata = [None]*16

for ct, idate in enumerate(udates):
    if diffs:
        if not makemovie and ct > 1:
            break
    else: 
        if not makemovie and ct > 0:
            break
    
    use = np.where(dates == idate)[0]
    use = use[np.argsort(files[use])]
    
    #cm = matplotlib.cm.get_cmap(cmap)
    #bkcol = cm(0.)
    
    
    figsizes = {720: (8.54, 4.8), 1080: (19.2, 10.8)}
    
    #f2 = plt.figure(2, figsize=figsizes[reso])
    # histbins = np.arange(0, 1e4, 20)
    
    fig = plt.figure(1, figsize=figsizes[reso], frameon=False)
    # make the plot cover the entire figure with the right background colors
    if single:
        ax = fig.add_axes([slx, 0+pad, 9/16*(1-2*pad), 1-2*pad])
    else:
        ax = fig.add_axes([0.0, 0, 1, 1])
    ax.axis('off')
    fig.patch.set_facecolor(bkcol)
    ax.patch.set_facecolor(bkcol)
    
    ytxtoff = 0.0
    
    for ii, iuse in enumerate(use):
        with fits.open(files[iuse]) as ff:
            #print(ff[1].header['camera'], ff[1].header['ccd'], ff[1].header['crval1'], ff[1].header['crval2'])
            ind = (int(ff[1].header['camera'])-1) * 4 + int(ff[1].header['ccd']) - 1
            
            #wcs = WCS(ff[1].header)
            data = ff[1].data * 1
            if not diffs:
                data[data < 0.9*vmin] = 0.9*vmin
    
            if flips[ind]:
                extent = (xlocs[ind]+1-gap, xlocs[ind]+gap, ylocs[ind]+1-gap, ylocs[ind]+gap)
            else:
                extent = (xlocs[ind]+gap, xlocs[ind]+1-gap, ylocs[ind]+gap, ylocs[ind]+1-gap)
    
            if diffs:
                if olddata[ind] is not None:
                    cbpl = plt.imshow(data.T - olddata[ind].T, vmin=vmin, vmax=vmax, # norm=colors.LogNorm(vmin=vmin, vmax=vmax)
                                      extent=extent, cmap=diffcmap)
            else:    
                cbpl = plt.imshow(data.T, norm=colors.LogNorm(vmin=vmin, vmax=vmax), #vmin=vmin, vmax=vmax,
                                  extent=extent, cmap=cmap)
            if single:
                lw = 6
            else:
                lw = 1
            plt.plot([extent[0], extent[0]], [extent[2], extent[3]], color='white', lw=lw)
            plt.plot([extent[1], extent[1]], [extent[2], extent[3]], color='white', lw=lw)
            plt.plot([extent[0], extent[1]], [extent[2], extent[2]], color='white', lw=lw)
            plt.plot([extent[0], extent[1]], [extent[3], extent[3]], color='white', lw=lw)
            
            txt = 'Cam {0}\nCCD {1}'.format(ff[1].header['camera'], ff[1].header['ccd'])
            
            if ylocs[ind] == 1:
                ytxt = 2 + ytxtoff
                va = 'bottom'
            else:
                ytxt = 0 - ytxtoff - 0.01
                va = 'top'
                
            if not single:
                plt.text(xlocs[ind] + 0.5, ytxt, txt, ha='center', va=va, color=fontcol, fontproperties=prop, fontsize=fszs4[reso])
    
            tstart = datetime.strptime(ff[0].header['date-obs'].split('.')[0],
                                        '%Y-%m-%dT%H:%M:%S')
            tend = datetime.strptime(ff[0].header['date-end'].split('.')[0],
                                        '%Y-%m-%dT%H:%M:%S')
            tmid = tstart + (tend - tstart)/2
            
            #plt.figure(2)
            #txt2 = 'Cam {0}; CCD {1}'.format(ff[1].header['camera'], ff[1].header['ccd'])
            #plt.hist(data.flatten(), bins=histbins, label=txt2, alpha=0.3, histtype='step')
            #plt.figure(1)
            olddata[ind] = data
    
    if len(use) == 0:
        olddata = [None]*16

    if (ct % 20) == 0:
        print('Image {0} of {1}'.format(ct+1, len(udates)))
    
    if diffs and ct == 0:
        plt.close(fig)
        continue
    
    if len(use) == 0:
        tmid = datetime.strptime(idate, '%Y%j%H%M%S') + delt/2
        if single:
            plt.plot([extent[0], extent[0]], [extent[2], extent[3]], color='white', lw=lw)
            plt.plot([extent[1], extent[1]], [extent[2], extent[3]], color='white', lw=lw)
            plt.plot([extent[0], extent[1]], [extent[2], extent[2]], color='white', lw=lw)
            plt.plot([extent[0], extent[1]], [extent[3], extent[3]], color='white', lw=lw)
            plt.text(slx + 9/32*(1-2*pad), 0.5, 'Data\nDownlink\nGap',transform=fig.transFigure,
                     ha='center', va='center', color=fontcol, fontproperties=prop, fontsize=fszs5[reso])
        else:
            plt.text(0.5, 0.5, 'Data\nDownlink\nGap',transform=fig.transFigure,
                     ha='center', va='center', color=fontcol, fontproperties=prop, fontsize=fszs5[reso])
        
    if not single:
        plt.xlim(-0.14,8.14)
        plt.ylim(-0.05,2.05)
    else:
        if flips[ind]:
            plt.xlim(extent[1], extent[0])
            plt.ylim(extent[3], extent[2])
        else:
            plt.xlim(extent[0], extent[1])
            plt.ylim(extent[2], extent[3])
    
    yshift = 0.1
    if not single:
        titletxt = plt.text(0.5, 0.99, titlestr, transform=fig.transFigure,
                            ha='center', va='top', color=fontcol, fontproperties=prop, fontsize=fszs2[reso])
        plt.text(0.395, 0.86, tmid.strftime('%d %b %Y %H:%M'), transform=fig.transFigure,
                 ha='left', va='top', color=fontcol, fontproperties=prop, fontsize=fszs3[reso])
    else:
        titletxt = plt.text(slx/2, 0.99-yshift, titlestr, transform=fig.transFigure,
                            ha='center', va='top', color=fontcol, fontproperties=prop, fontsize=fszs2[reso])
        if diffs:
            plt.text(slx/2-0.105, 0.73-yshift-.05, tmid.strftime('%d %b %Y %H:%M'), transform=fig.transFigure,
                     ha='left', va='top', color=fontcol, fontproperties=prop, fontsize=fszs3[reso])
        else:
            plt.text(slx/2-0.105, 0.73-yshift, tmid.strftime('%d %b %Y %H:%M'), transform=fig.transFigure,
                     ha='left', va='top', color=fontcol, fontproperties=prop, fontsize=fszs3[reso])
    
    if single:
        # top neighbor
        if ylocs[ind] == 0:
            neighb = np.where((xlocs == xlocs[ind]) & (ylocs == 1))[0][0]
            ncam = neighb // 4 + 1
            nccd = (neighb % 4) + 1
            ntxt = 'Cam {0}     CCD {1}'.format(ncam, nccd)
            plt.text(slx + 9/32*(1-2*pad), 1-pad/2,ntxt,transform=fig.transFigure,
                     ha='center', va='center', color=fontcol, fontproperties=prop, fontsize=fszs1[reso])
            ax.annotate("",
            xy=(slx + 9/32*(1-2*pad), 1-pad/4), xycoords='figure fraction',
            xytext=(slx + 9/32*(1-2*pad), 1-pad*3/4), textcoords='figure fraction',
            arrowprops=dict(arrowstyle="wedge, tail_width=2, shrink_factor=0.5", color=fontcol), annotation_clip=False,
            ha='center', va='center',
            color=fontcol)
            
        else:
            # bottom neighbor
            neighb = np.where((xlocs == xlocs[ind]) & (ylocs == 0))[0][0]
            ncam = neighb // 4 + 1
            nccd = (neighb % 4) + 1
            ntxt = 'Cam {0}     CCD {1}'.format(ncam, nccd)
            plt.text(slx + 9/32*(1-2*pad), pad/2,ntxt,transform=fig.transFigure,
                     ha='center', va='center', color=fontcol, fontproperties=prop, fontsize=fszs1[reso])
            ax.annotate("",
            xy=(slx + 9/32*(1-2*pad), pad/4), xycoords='figure fraction',
            xytext=(slx + 9/32*(1-2*pad), pad*3/4), textcoords='figure fraction',
            arrowprops=dict(arrowstyle="wedge, tail_width=2, shrink_factor=0.5", color=fontcol), annotation_clip=False,
            ha='center', va='center',
            color=fontcol)
        # right neighbor
        neighb = np.where((xlocs == xlocs[ind] + 1) & (ylocs == ylocs[ind]))[0]
        if len(neighb) > 0:
            if len(neighb) > 1:
                sys.exit(1)
            neighb = neighb[0]
            ncam = neighb // 4 + 1
            nccd = (neighb % 4) + 1
            ntxt = 'Cam{0}\nCCD{1}'.format(ncam, nccd)
            ntxtstr = ''
            for char in ntxt:
                ntxtstr += char + '\n'
            ntxtstr = ntxtstr[:-1]
            plt.text(slx + 9/16*(1-2*pad)+pad/2*9/16, 0.5, ntxtstr, transform=fig.transFigure,
                     ha='center', va='center', color=fontcol, fontproperties=prop, fontsize=fszs1[reso],
                     multialignment='center')
            ax.annotate("",
            xy=(slx + 9/16*(1-2*pad)+pad*3/4*9/16, 0.5), xycoords='figure fraction',
            xytext=(slx + 9/16*(1-2*pad)+pad/4*9/16, 0.5), textcoords='figure fraction',
            arrowprops=dict(arrowstyle="wedge, tail_width=2, shrink_factor=0.5", color=fontcol), annotation_clip=False,
            ha='center', va='center',
            color=fontcol)
        # left neighbor
        neighb = np.where((xlocs == xlocs[ind] - 1) & (ylocs == ylocs[ind]))[0]
        if len(neighb) > 0:
            if len(neighb) > 1:
                sys.exit(1)
            neighb = neighb[0]
            ncam = neighb // 4 + 1
            nccd = (neighb % 4) + 1
            ntxt = 'Cam{0}\nCCD{1}'.format(ncam, nccd)
            ntxtstr = ''
            for char in ntxt:
                ntxtstr += char + '\n'
            ntxtstr = ntxtstr[:-1]
            plt.text(slx - pad/2*9/16, 0.5, ntxtstr, transform=fig.transFigure,
                     ha='center', va='center', color=fontcol, fontproperties=prop, fontsize=fszs1[reso],
                     multialignment='center')
            ax.annotate("",
            xy=(slx - pad*3/4*9/16, 0.5), xycoords='figure fraction',
            xytext=(slx - pad/4*9/16, 0.5), textcoords='figure fraction',
            arrowprops=dict(arrowstyle="wedge, tail_width=2, shrink_factor=0.5", color=fontcol), annotation_clip=False,
            ha='center', va='center',
            color=fontcol)
            
    # too lazy to figure out how to do this right
    lstr = ''
    for char in leftlabel:
        lstr += char + '\n'
    lstr = lstr[:-1]
    
    rstr = ''
    for char in rightlabel:
        rstr += char + '\n'
    rstr = rstr[:-1]
    
    if not single and len(use) > 0:
        plt.text(0.002, 0.5, lstr, transform=fig.transFigure,
                 ha='left', va='center', multialignment='center', color=fontcol, fontproperties=prop, fontsize=fszs1[reso])
        plt.text(0.999, 0.5, rstr, transform=fig.transFigure,
                 ha='right', va='center', multialignment='center', color=fontcol, fontproperties=prop, fontsize=fszs1[reso])
    
    # plot my name
    plt.text(0.01, 0.01, credit, transform=fig.transFigure,
             ha='left', va='bottom', multialignment='left', color=fontcol, fontproperties=prop, fontsize=fszs4[reso])
    
    if not single:
        ax2 = fig.add_axes([0.35, 0.1, 0.3, 0.05])
    else:
        ax2 = fig.add_axes([(slx-0.3)/2, 0.2, 0.3, 0.05])
    
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
    
    cbar = plt.colorbar(cbpl, orientation='horizontal', cax=ax2, extend='both')
    if diffs:
        cbar.set_label('Change in Flux', color=fontcol, fontproperties=prop, fontsize=fszs1[reso])
    else:
        cbar.set_label('Calibrated Flux', color=fontcol, fontproperties=prop, fontsize=fszs1[reso])
    cbar.outline.set_edgecolor(fontcol)
    cbar.outline.set_linewidth(2)
    cbar.set_ticks(ticker.LogLocator(), update_ticks=True)
    # for some reason need to clear the original ticks because I can't figure out a way to 
    # change or remove them properly.
    cbar.ax.xaxis.set_ticks([])
    cbar.ax.xaxis.set_ticks([], minor=True)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabs)
    cbar.ax.tick_params(color=fontcol)
    cbar.ax.set_xticklabels(ticklabs, color=fontcol, fontproperties=prop, fontsize=fszs4[reso], horizontalalignment='center')
    cbar.ax.tick_params(axis='x', color=fontcol, width=2, length=5, zorder=5) # , length=5
    # why do these minor ticks keep coming back as little black overlays
    cbar.ax.xaxis.set_ticks([], minor=True)
    
    outstr = os.path.join(outdir, 'img{0:05d}.png'.format(ct))
    # XXX: figure out what the pixel flux is for stars of various magnitudes?
    plt.savefig(outstr, facecolor=fig.get_facecolor(), edgecolor='none')
        
    if makemovie:
        plt.close(fig) 
        

#plt.figure(2)
#plt.legend()
#plt.ylim(0.5, plt.ylim()[1])
#plt.yscale('log')
        

