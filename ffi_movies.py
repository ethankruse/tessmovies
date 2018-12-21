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
#cd = os.path.abspath(os.path.dirname(__file__))
cd = '/data/tessraid/ekruse1/tessmovies'

sector = 1
dataloc = '/data/tessraid/data/ffis/sector1'

outdir = '/data/tessraid/ekruse1/tessmovies/movies/sector1/all'

cmap = 'viridis'
#cmap = 'plasma'
#cmap = 'gray'

fontcol = 'white'
fontfile = os.path.join(cd, 'Avenir-Black.otf')

titlestr = 'TESS: The Movie\nSector {0}'.format(sector)

leftlabel = 'ECLIPTIC'
rightlabel = 'SOUTH POLE'

credit = 'By Ethan Kruse\n@ethan_kruse'

reso = 1080

# font sizes at various resolutions
# 1280x720, 1920x1080
fszs1 = {720: 12, 1080: 20}
fszs2 = {720: 18, 1080: 30}
fszs3 = {720: 15, 1080: 25}
fszs4 = {720: 9, 1080: 15}

makemovie = True

# background color
bkcol = 'black'






if not os.path.exists(outdir):
    os.mkdir(outdir)
    
if makemovie:
    existing = glob(os.path.join(outdir, '*png'))
    for iexist in existing:
        os.remove(iexist)

prop = fm.FontProperties(fname=fontfile)

plt.close('all')

paths = os.path.join(dataloc, '*-s{0:04d}-*ffic.fits'.format(sector))
files = glob(paths)
files = np.array(files)
dates = []
for ifile in files:
    dates.append(os.path.split(ifile)[1][4:].split('-')[0])

dates = np.array(dates)
udates = np.unique(dates)



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

xlocs = [1, 1, 0, 0, 3, 3, 2, 2, 4, 4, 5, 5, 6, 6, 7, 7]
ylocs = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]
flips = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
gap = 0.02

vmin = 70
vmax = 1001.



for ct, idate in enumerate(udates):
    
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
            data[data < 0.9*vmin] = 0.9*vmin
    
            if flips[ind]:
                extent = (xlocs[ind]+1-gap, xlocs[ind]+gap, ylocs[ind]+1-gap, ylocs[ind]+gap)
            else:
                extent = (xlocs[ind]+gap, xlocs[ind]+1-gap, ylocs[ind]+gap, ylocs[ind]+1-gap)
    
            plt.imshow(data.T, norm=colors.LogNorm(vmin=vmin, vmax=vmax), #vmin=vmin, vmax=vmax,
                       extent=extent, cmap=cmap)
            plt.plot([extent[0], extent[0]], [extent[2], extent[3]], color='white')
            plt.plot([extent[1], extent[1]], [extent[2], extent[3]], color='white')
            plt.plot([extent[0], extent[1]], [extent[2], extent[2]], color='white')
            plt.plot([extent[0], extent[1]], [extent[3], extent[3]], color='white')
            
            txt = 'Cam {0}\nCCD {1}'.format(ff[1].header['camera'], ff[1].header['ccd'])
            
            if ylocs[ind] == 1:
                ytxt = 2 + ytxtoff
                va = 'bottom'
            else:
                ytxt = 0 - ytxtoff - 0.01
                va = 'top'
    
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
            
    plt.xlim(-0.14,8.14)
    plt.ylim(-0.05,2.05)
    
    titletxt = plt.text(0.5, 0.99, titlestr, transform=fig.transFigure,
                        ha='center', va='top', color=fontcol, fontproperties=prop, fontsize=fszs2[reso])
    plt.text(0.4, 0.86, tmid.strftime('%d %b %Y %H:%M'), transform=fig.transFigure,
             ha='left', va='top', color=fontcol, fontproperties=prop, fontsize=fszs3[reso])
    
    # too lazy to figure out how to do this right
    lstr = ''
    for char in leftlabel:
        lstr += char + '\n'
    lstr = lstr[:-1]
    
    rstr = ''
    for char in rightlabel:
        rstr += char + '\n'
    rstr = rstr[:-1]
    
    plt.text(0.002, 0.5, lstr, transform=fig.transFigure,
             ha='left', va='center', multialignment='center', color=fontcol, fontproperties=prop, fontsize=fszs1[reso])
    plt.text(0.998, 0.5, rstr, transform=fig.transFigure,
             ha='right', va='center', multialignment='center', color=fontcol, fontproperties=prop, fontsize=fszs1[reso])
    
    # plot my name
    plt.text(0.01, 0.01, credit, transform=fig.transFigure,
             ha='left', va='bottom', multialignment='left', color=fontcol, fontproperties=prop, fontsize=fszs4[reso])
    
    ax2 = fig.add_axes([0.35, 0.1, 0.3, 0.05])
    
    ticks = np.arange((vmin//100)*100 + 100, vmax, 100).astype(int)
    ticklabs = []
    for itick in ticks:
        ticklabs.append(str(itick))
    for ii in np.arange(1, len(ticklabs)-1):
        ticklabs[ii] = ''
    
    cbar = plt.colorbar(orientation='horizontal', cax=ax2, extend='both')
    cbar.set_label('Calibrated Flux', color=fontcol, fontproperties=prop, fontsize=fszs1[reso])
    cbar.outline.set_edgecolor(fontcol)
    cbar.outline.set_linewidth(2)
    
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabs)
    cbar.ax.tick_params(color=fontcol)
    cbar.ax.set_xticklabels(ticklabs, color=fontcol, fontproperties=prop, fontsize=fszs4[reso], horizontalalignment='center')
    cbar.ax.tick_params(axis='x', color=fontcol, width=2, length=5) # , length=5
    
    outstr = os.path.join(outdir, 'img{0:05d}.png'.format(ct))
    # XXX: figure out what the pixel flux is for stars of various magnitudes?
    plt.savefig(outstr, facecolor=fig.get_facecolor(), edgecolor='none')

    if (ct % 20) == 0:
        print('Image {0} of {1}'.format(ct+1, len(udates)))
        
    if makemovie:
        plt.close(fig) 

#plt.figure(2)
#plt.legend()
#plt.ylim(0.5, plt.ylim()[1])
#plt.yscale('log')
