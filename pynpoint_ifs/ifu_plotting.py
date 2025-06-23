from scipy.ndimage import shift
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_ticks(img,star_pos,pixscale,tick_sep=0.5,minor_tick_sep = 0.1):
    lenx=len(img)
    ticks_x = np.arange(lenx)
    ticks_y = np.arange(lenx)
    
    sky_x = -(ticks_x-star_pos[0])*pixscale
    bound_x = int(np.max(np.abs(sky_x)//tick_sep)+1)*tick_sep
    nb_ticks = int(np.round((2*bound_x)/tick_sep) + 1)
    ticks_x_labels = np.linspace(-bound_x,bound_x,nb_ticks)
    tick_x_pos = (-ticks_x_labels)/pixscale + star_pos[0]
    
    nb_ticks = int(np.round((2*bound_x)/minor_tick_sep) + 1)
    ticks_x_labels_minor = np.linspace(-bound_x,bound_x,nb_ticks)
    tick_x_pos_minor = (-ticks_x_labels_minor)/pixscale + star_pos[0]
    
    sky_y = (ticks_y-star_pos[1])*pixscale
    bound_y = int(np.max(np.abs(sky_y)//tick_sep)+1)*tick_sep
    nb_ticks = int(np.round((2*bound_y)/tick_sep) + 1)
    ticks_y_labels = np.linspace(-bound_y,bound_y,nb_ticks)
    tick_y_pos = (ticks_y_labels)/pixscale + star_pos[1]
    
    nb_ticks = int(np.round((2*bound_y)/minor_tick_sep) + 1)
    ticks_y_labels_minor = np.linspace(-bound_y,bound_y,nb_ticks)
    tick_y_pos_minor = (ticks_y_labels_minor)/pixscale + star_pos[1]
    
    return ticks_x_labels,tick_x_pos,ticks_x_labels_minor,tick_x_pos_minor,ticks_y_labels,tick_y_pos,ticks_y_labels_minor,tick_y_pos_minor

def plot_circle(ax,position,radius,color='w'):
    c1 = plt.Circle((position[0],position[1]),radius,facecolor='none',edgecolor=color,ls='dashed')
    ax.add_patch(c1)
def annotate_snr(ax,position,snr,fontsize=10):
    if snr >= 100:
        ax.annotate('S/N=%i' % (int(np.round(snr))),(position[0],position[1]),color='w',fontsize=fontsize,weight='bold',horizontalalignment='center')
    else:
        ax.annotate('S/N=%.1f' % (snr),(position[0],position[1]),color='w',fontsize=fontsize,weight='bold',horizontalalignment='center')
def make_ccf_plot(ax,cc,drv,rv_planet,planet_pos,star_pos,vmin=0,vmax=5):
    rv_max_i=np.argmin(np.abs(drv-rv_planet))
    img = cc[rv_max_i]
    img_norm = (img-np.nanmean(img))/np.nanstd(img)
    img_norm[np.isnan(img_norm)] = np.nanmedian(img_norm)-0.05*np.nanstd(img_norm)
    
    im = ax.imshow(img_norm,origin='lower',vmin=vmin,vmax=vmax,cmap='magma')
    
    ax.scatter(x= star_pos[0], y= star_pos[1],s=50,color='orange',marker='*')
    
    
    # colorbar
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = plt.colorbar(im, cax=cax)
    # cbar.set_label('CCF (standardised)', rotation=270,labelpad=10)
def make_snr_plot(ax,ccf,rv,std,signal_range=(-100,100),crop=5,vmin=0,vmax=8.5,tick_params = [],fontsize=12,star_pos = [],planet_pos = [],title='title',annotate=True):
    index_low = np.where(rv>=signal_range[0])[0][0]
    index_high = np.where(rv<=signal_range[1])[0][-1]
    alt_snr = np.nanmax(ccf[index_low:index_high,:,:],axis=0)/std
    # smooth = gaussian_filter(alt_snr,0.75)[crop:-crop,crop:-crop]
    smooth = alt_snr[crop:-crop,crop:-crop]
    # molecular map
    im = ax.imshow(smooth,origin='lower',vmin=vmin,vmax=vmax,cmap='magma')# cmap=cmap[mol])
    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('S/N', rotation=270,labelpad=10)
    # star position
    ax.scatter(x= star_pos[0], y= star_pos[1],s=50,color='orange',marker='*')
    # ticks and labels
    make_labels(ax,tick_params=tick_params,img=smooth,fontsize=fontsize,title=title)
    # snr annotation
    # ax.scatter(x=planet_pos[0],y=planet_pos[1])
    if annotate:
        # ax.annotate('%.1f' % (np.nanmax(smooth)),(planet_pos[0], 5 + planet_pos[1]),color='w',fontsize=fontsize,weight='bold')
        ax.annotate('S/N=%.1f' % (smooth[int(planet_pos[1]),int(planet_pos[0])]),(planet_pos[0], 3.5 + planet_pos[1]),color='w',fontsize=fontsize,weight='bold',horizontalalignment='center')

def make_PSF_plot(ax,image,crop=5,tick_params = [],fontsize=12,star_pos = [],title='title',norm='log'):
    image_mean_norm = image/np.nanmax(image)
    if norm=='log':
        norm=colors.LogNorm(vmin=np.nanpercentile(image_mean_norm,5), vmax=np.nanpercentile(image_mean_norm,100))
    else:
        norm=colors.Normalize(vmin=np.nanpercentile(image_mean_norm,5), vmax=np.nanpercentile(image_mean_norm,100))
    im = ax.imshow(image_mean_norm,origin='lower',norm = norm, cmap='afmhot')
    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    if norm=='log':
        cbar_label = 'Contrast [dex]'
        lowerval = 4
        ticks = [1./10**(i) for i in np.arange(lowerval)]
        ticklabels = np.arange(0,-lowerval,-1)
        cbar = plt.colorbar(im, cax=cax,ticks=ticks)
        cbar.ax.set_yticklabels(ticklabels)
    elif norm=='linear':
        cbar_label = 'Contrast (a.u.)'
        cbar = plt.colorbar(im, cax=cax)
    else:
        cbar_label = 'Contrast'
        cbar = plt.colorbar(im, cax=cax)
    
    cbar.set_label(cbar_label, rotation=270,labelpad=10)
    # star position
    ax.scatter(x= star_pos[0], y= star_pos[1],s=50,color='orange',marker='*')
    # ticks and labels
    make_labels(ax,tick_params=tick_params,img=image_mean_norm,fontsize=fontsize,title=title)

def make_labels(ax,tick_params,img,fontsize=12,title='title'):
    ticks_x_labels,tick_x_pos,ticks_x_labels_minor,tick_x_pos_minor,ticks_y_labels,tick_y_pos,ticks_y_labels_minor,tick_y_pos_minor=tick_params
    ax.set_xticks(ticks=tick_x_pos,labels=map(lambda x: '%.1f' % x,ticks_x_labels),fontsize=fontsize-4)
    ax.set_xticks(minor=True,ticks=tick_x_pos_minor,labels=map(lambda x: '%.1f' % x,ticks_x_labels_minor),fontsize=fontsize-4)
    ax.set_yticks(ticks=tick_y_pos,labels=map(lambda x: '%.1f' % x,ticks_y_labels),fontsize=fontsize-4)
    ax.set_yticks(minor=True,ticks=tick_y_pos_minor,labels=map(lambda x: '%.1f' % x,ticks_y_labels_minor),fontsize=fontsize-4)
    ax.tick_params(axis='both',which='minor',labelleft=False,labelbottom=False)
    ax.set_xlabel(r'$\Delta$RA ["]',fontsize=fontsize-2,labelpad=2)
    ax.set_ylabel(r'$\Delta$DEC ["]',fontsize=fontsize-2,labelpad=-2)
    ax.set_title(title,fontsize=fontsize,pad=4)
    ax.set_xlim((0,len(img)-1))
    ax.set_ylim((0,len(img)-1))
def nice_name(mol):
    if mol=='chem_equ':
        return 'Full model'
    result = ''
    for str in mol:
        if str == '_':
            break
        if np.char.isnumeric(str):
            result += '$_{%s}$' % str
        else:
            result += str
    if mol in ['CO_36','13CO','CO36']:
        result = '$^{13}$C$^{16}$O'
        
    return result

def nice_psf_image(datacubes,upscale,cubeposition):
    images = np.mean(datacubes,axis=1)
    lencube,lenx,leny = np.shape(images)
    images_upscaled = np.zeros((len(images),upscale*lenx,upscale*leny))
    for img_i in range(lencube):
        for i in range(upscale):
            for j in range(upscale):
                images_upscaled[img_i,i::upscale,j::upscale] = images[img_i,:,:]
    shifted_images = np.zeros_like(images_upscaled)
    for img_i in range(lencube):
        shifted_images[img_i,:,:] = shift(images_upscaled[img_i,:,:],(-upscale*cubeposition[img_i,1],-upscale*cubeposition[img_i,0]))
    return shifted_images

def combine_psf_image(shifted_images):
    mask_image = np.abs(shifted_images) < 0.0001
    mask_image_reduced = np.zeros_like(mask_image)
    for cube_i in range(len(mask_image)):
        mask_image_reduced[cube_i,1:,1:] = np.min((np.dstack([
            mask_image[cube_i,0:-1,0:-1],
            mask_image[cube_i,1:,0:-1],
            mask_image[cube_i,1:,1:],
            mask_image[cube_i,0:-1,1:]
        ])),axis=2)
    mask_image_reduced[:,0,:] = mask_image[:,0,:]
    mask_image_reduced[:,:,0] = mask_image[:,:,0]
    
    nb_images = np.sum(mask_image,axis=0)
    nb_images = nb_images/np.max(nb_images)
    nb_images_smooth = gaussian_filter(nb_images,2)
    nb_images_smooth = nb_images_smooth/np.max(nb_images_smooth)
    
    nb_images_smooth_v2 = gaussian_filter(nb_images,1.5)
    mask_common = nb_images_smooth_v2==np.max(nb_images_smooth_v2)
    flux_common = np.median(shifted_images[:,mask_common],axis=1)
    
    image_mean = np.nansum(shifted_images*mask_image_reduced/flux_common[:,np.newaxis,np.newaxis]/nb_images,axis=0)
    
    return image_mean