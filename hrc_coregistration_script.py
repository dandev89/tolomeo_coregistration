'''
Created on 11/set/2014
Last modified on 24/oct/2014

@author: Daniele De Vecchi

Universita' degli Studi di Pavia
TOLOMEO project

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
'''

import os,sys
import config
import numpy as np
import scipy as sp
from scipy import ndimage
import osgeo.gdal
import osgeo.ogr
from gdalconst import *
import otbApplication
import cv2
import time 
import collections
import subprocess
import random
import shutil
import argparse
from operator import itemgetter, attrgetter
from numpy.fft import fft2, ifft2, fftshift
data_type = np.uint8
from library import *
sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])


def main():
    arg = args()
    hrc_folder = str(arg.hrc_folder)
    ccd_folder = str(arg.ccd_folder)
    compute_ndvi = arg.compute_ndvi
    hrc_coregistration(hrc_folder,ccd_folder,compute_ndvi)

def args():
    parser = argparse.ArgumentParser(description='HRC Co-Registration')
    parser.add_argument("hrc_folder", help="Folder with HRC images. If you're using Windows please use '/' or double '\\' as separator")
    parser.add_argument("ccd_folder", help="Folder with CCD images. If you're using Windows please use '/' or double '\\' as separator")
    parser.add_argument("--compute_ndvi", default=False, const=True, nargs='?', help="Enable calculation of the NDVI index from the fixed images")
    args = parser.parse_args()
    return args

def hrc_coregistration(hrc_folder,ccd_folder,compute_ndvi):
    start_time = time.time()
    #Folder for HRCs
    #Folder for CCDs -> Loop through the CCD folder and determine the HRCs intersecting it (create a list with names)
    
    hrc_files = os.listdir(hrc_folder)
    hrc_files = [f for f in hrc_files if "HRC" in f and (".tif" in f or ".TIF" in f) and "xml" not in f]
    print hrc_files
    ccd_files = os.listdir(ccd_folder)
    ccd_files = [f for f in ccd_files if "CCD" in f and (".tif" in f or ".TIF" in f) and "xml" not in f]
    print ccd_files
    
    path_row_list_ccd = []
    for f in ccd_files:
        f_split = f.split('_')
        path_row_list_ccd.append((f_split[4],f_split[5]))
    path_row_list_ccd = list(set(path_row_list_ccd)) #extract unique values
    print path_row_list_ccd
    
    path_row_list_hrc = []
    for f in hrc_files:
        f_split = f.split('_')
        path_row_list_hrc.append((f_split[4],f_split[6]))
    path_row_list_hrc = list(set(path_row_list_hrc)) #extract unique values
    print path_row_list_hrc
    

    #Adjust every CCD image provided
    for f in range(0,len(path_row_list_ccd)):
        print '{} of {}'.format(str(f+1),str(len(path_row_list_ccd)))
        path = path_row_list_ccd[f][0]
        row = path_row_list_ccd[f][1]
        target_files = [str(ccd_folder) + '\\' + str(c) for c in ccd_files if (".tif" in c or ".TIF" in c) and "xml" not in c and "CCD" in c and "rpj" not in c and "BAND5" not in c and str(path_row_list_ccd[f][0]) in c and str(path_row_list_ccd[f][1]) in c]
        band2_file = [str(ccd_folder) + '\\' + str(c) for c in ccd_files if (".tif" in c or ".TIF" in c) and "xml" not in c and "CCD" in c and "rpj" not in c and "BAND5" not in c and str(path_row_list_ccd[f][0]) in c and str(path_row_list_ccd[f][1]) in c and "BAND2" in c]
        band3_file = [str(ccd_folder) + '\\' + str(c) for c in ccd_files if (".tif" in c or ".TIF" in c) and "xml" not in c and "CCD" in c and "rpj" not in c and "BAND5" not in c and str(path_row_list_ccd[f][0]) in c and str(path_row_list_ccd[f][1]) in c and "BAND3" in c]
        band4_file = [str(ccd_folder) + '\\' + str(c) for c in ccd_files if (".tif" in c or ".TIF" in c) and "xml" not in c and "CCD" in c and "rpj" not in c and "BAND5" not in c and str(path_row_list_ccd[f][0]) in c and str(path_row_list_ccd[f][1]) in c and "BAND4" in c]
        ref_files = [str(hrc_folder) + '\\' + str(c) for c in hrc_files if (".tif" in c or ".TIF" in c) and "xml" not in c and "HRC" in c and "rpj" not in c and str(path_row_list_ccd[f][0]) in c[22:] and str(path_row_list_ccd[f][1]) in c[22:]]

        print target_files
        print band2_file
        print ref_files
        #ref_files = ref_files[:3] #limit the analysis to the first 3 HRC files found
        if ref_files:
            rand_k = 5
            if len(ref_files) < rand_k:
                rand_k = len(ref_files)
            rand_selection = random.sample(range(0,len(ref_files)),rand_k)
            #print rand_selection
            ref_files = [ref_files[rs] for rs in rand_selection]
            #print ref_files
            ref_files = list(set(ref_files)) #unique files to avoid repetition
            
            print 'Processing...' + str(path_row_list_ccd[f][0]) + ', ' + str(path_row_list_ccd[f][1])
            
            layer_stack(target_files,ccd_folder+'\\multi.tif',data_type)
            unsupervised_classification_otb(ccd_folder+'\\multi.tif',ccd_folder+'\\multi_class.tif',5,10)
            
            target_tiles_list = []
            ref_tiles_list = []
            counter = 0
            for image_ref in ref_files:
                minx_ref,miny_ref,maxx_ref,maxy_ref = get_coordinate_limit(image_ref)
                rows_ref_orig,cols_ref_orig,nbands_ref_orig,geotransform_ref_orig,projection_ref_orig = read_image_parameters(image_ref)
                
                image_target = band2_file[0]
                minx_target,miny_target,maxx_target,maxy_target = get_coordinate_limit(image_target)
                rows_target_orig,cols_target_orig,nbands_target_orig,geotransform_target_orig,projection_target_orig = read_image_parameters(image_target)
    
                minx = np.max([minx_target,minx_ref])
                miny = np.max([miny_target,miny_ref])
                maxx = np.min([maxx_target,maxx_ref])
                maxy = np.min([maxy_target,maxy_ref])

                stat_list = classification_statistics(ccd_folder+'\\multi_class.tif',band2_file[0])
                stat_list = [element for element in stat_list if element[2]!=255 and element[5] < 20.0 and element[6] < 20.0 and element[7] > 1500000]
                stat_list_sorted = sorted(stat_list,key=itemgetter(7))
                stat_list_sorted = sorted(stat_list_sorted,key=itemgetter(4,5,6))
                #print stat_list_sorted
                
                class_iteration = False
                class_def = 0

                while class_iteration == False:
                    try:
                        selected_class = stat_list_sorted[class_def][0]
                        force_class = False
                    except:
                        #class_def = class_def - 1
                        selected_class = stat_list_sorted[class_def-1][0]
                        force_class = True
    
                    class_list,start_col_coord,start_row_coord,end_col_coord,end_row_coord = extract_tiles(ccd_folder+'\\multi_class.tif',minx,maxy,maxx,miny,data_type)
                    target_mask = np.equal(class_list,selected_class)
                    new_mask = sp.ndimage.binary_fill_holes(target_mask, structure=None, output=None, origin=0)
                    new_mask = sp.ndimage.binary_opening(new_mask, structure=np.ones((21,21))).astype(np.int)
                    
                    #New mask statistics
                    new_mask_flat = new_mask.flatten()
                    new_mask_counter = collections.Counter(new_mask_flat).most_common()
                    #print new_mask_counter
                    count_list = [count for elt,count in new_mask_counter]
                    #print count_list
                    if force_class == False:
                        if count_list:
                            try:
                                zeros_perc = (float(count_list[0]) / float(count_list[0]+count_list[1]))*100
                            except:
                                zeros_perc = 0
                            try:
                                ones_perc = (float(count_list[1]) / float(count_list[0]+count_list[1]))*100
                            except:
                                ones_perc = 0
                        else:
                            class_iteration = False
                            class_def = class_def + 1

                        if ones_perc > 15.0: 
                            class_iteration = True
                        else:
                            class_iteration = False
                            class_def = class_def +1
                    else:
                        class_iteration = True
                
                
                band_mat_target,start_col_coord_target,start_row_coord_target,end_col_coord_target,end_row_coord_target = extract_tiles(band2_file[0],minx,maxy,maxx,miny,data_type)
                resampling(image_ref,image_ref[:-4]+'_rs_20.tif',20.0,'bicubic') #downsample 
                image_ref = image_ref[:-4]+'_rs_20.tif'
                band_mat_ref,start_col_coord_ref,start_row_coord_ref,end_col_coord_ref,end_row_coord_ref = extract_tiles(image_ref,minx,maxy,maxx,miny,data_type)
                
                geotransform_cut = [minx,geotransform_target_orig[1],0.0,maxy,0.0,geotransform_target_orig[5]]
                rows_cut,cols_cut = band_mat_target.shape
                write_image([new_mask],data_type,0,band2_file[0][:-4]+'_mask.tif',rows_cut,cols_cut,geotransform_cut,projection_target_orig)
                rast2shp(band2_file[0][:-4]+'_mask.tif',band2_file[0][:-4]+'_mask.shp')
                split_shape(band2_file[0][:-4]+'_mask.shp',option="file",output_shape=band2_file[0][:-4]+'_mask_final.shp')
                clip_rectangular(band2_file[0],data_type,band2_file[0][:-4]+'_mask_final.shp',band2_file[0][:-4]+'_cut_'+str(counter)+'.tif',option="from_class")
                target_tiles_list.append(band2_file[0][:-4]+'_cut_'+str(counter)+'.tif')
                clip_rectangular(image_ref,data_type,band2_file[0][:-4]+'_mask_final.shp',image_ref[:-4]+'_cut_'+str(counter)+'.tif',option="from_class")
                ref_tiles_list.append(image_ref[:-4]+'_cut_'+str(counter)+'.tif')
                
                os.remove(image_ref)
                os.remove(band2_file[0][:-4]+'_mask.tif')
                os.remove(band2_file[0][:-4]+'_mask.shp')
                os.remove(band2_file[0][:-4]+'_mask.shx')
                os.remove(band2_file[0][:-4]+'_mask.dbf')
                os.remove(band2_file[0][:-4]+'_mask.prj')
                os.remove(band2_file[0][:-4]+'_mask_final.shp')
                os.remove(band2_file[0][:-4]+'_mask_final.shx')
                os.remove(band2_file[0][:-4]+'_mask_final.dbf')
                os.remove(band2_file[0][:-4]+'_mask_final.prj')
                
                counter = counter + 1

            for f in range(0,len(ref_tiles_list)):
                resampling(ref_tiles_list[f],ref_tiles_list[f][:-4]+'_rs_20.tif',20.0,'bicubic')
                ref_tiles_list[f] = ref_tiles_list[f][:-4]+'_rs_20.tif'
            b = [os.remove(c[:-10]+'.tif') for c in ref_tiles_list]
    
            ext_points_list = []
            x_shift_list = []
            y_shift_list = []
            #print target_tiles_list
            for f in range(0,len(target_tiles_list)):
                #print target_tiles_list[f]
                #print ref_tiles_list[f]
                band_list_target = read_image(target_tiles_list[f],data_type,0)
                band_list_ref = read_image(ref_tiles_list[f],data_type,0)
                ro,co,nb,gt_ref,proj_ref = read_image_parameters(ref_tiles_list[f])
                ro,co,nb,gt_target,proj_target = read_image_parameters(target_tiles_list[f])
                
                try:
                    kp_ref,kp_target,ext_points = EUC_SURF(band_list_ref[0],band_list_target[0],output_as_array=True)
                    ext_points_list.append(ext_points)
                except:
                    print 'OpenCV error'
            
            a = [os.remove(f) for f in target_tiles_list]
            b = [os.remove(c) for c in ref_tiles_list]
            
            x_shift, y_shift = 0,0
            x_shift_list = []
            y_shift_list = []

            for f in range(0,len(ext_points_list)):
                for p in range(0,len(ext_points_list[f])):
                    x_ref,y_ref,x_target,y_target = int(ext_points_list[f][p][0]),int(ext_points_list[f][p][1]),int(ext_points_list[f][p][2]),int(ext_points_list[f][p][3])
                    x_shift_list.append(float(x_ref-x_target))
                    y_shift_list.append(float(y_ref-y_target))
            
            x_shift_counter = collections.Counter(x_shift_list)
            x_shift_common = x_shift_counter.most_common()
            print x_shift_common
            #pick the first two values in frequency
            if x_shift_common[0][1] > 20:
                x_shift_best_1 =  x_shift_common[0][0] 
                x_shift_best_2 = x_shift_common[1][0]
                x_shift_filt_1 = [elt for elt,count in x_shift_common if abs(elt-x_shift_best_1) < 15]
                x_shift_filt_2 = [elt for elt,count in x_shift_common if abs(elt-x_shift_best_2) < 15]
                if len(x_shift_filt_1) > len(x_shift_filt_2):
                    x_shift_filt = x_shift_filt_1
                else:
                    x_shift_filt = x_shift_filt_2
                
            else:
                x_shift_filt = []
                for xp in range(0,len(x_shift_common)):
                    x_shift_ist = [x for x in x_shift_common if abs(x[0]-x_shift_common[xp][0]) < 15]
                    if len(x_shift_ist) > len(x_shift_filt):
                        x_shift_filt = x_shift_ist
            print x_shift_filt
            x_shift_weight = [elt*count for elt,count in x_shift_filt]
            x_freq_list = [count for elt,count in x_shift_filt]
            x_divider = np.sum(x_freq_list)
            y_shift_counter = collections.Counter(y_shift_list)
            y_shift_common = y_shift_counter.most_common()
            print y_shift_common
            #pick the first two values in frequency
            if y_shift_common[0][1] > 20:
                y_shift_best_1 = y_shift_common[0][0]
                y_shift_best_2 = y_shift_common[1][0]
                y_shift_filt_1 = [elt for elt,count in y_shift_common if abs(elt-y_shift_best_1) < 15]
                y_shift_filt_2 = [elt for elt,count in y_shift_common if abs(elt-y_shift_best_2) < 15]
                if len(y_shift_filt_1) > len(y_shift_filt_2):
                    y_shift_filt = y_shift_filt_1
                else:
                    y_shift_filt = y_shift_filt_2
                
            else:
                y_shift_filt = []
                for yp in range(0,len(y_shift_common)):
                    y_shift_ist = [y for y in y_shift_common if abs(y[0]-y_shift_common[yp][0]) < 15]
                    if len(y_shift_ist) > len(y_shift_filt):
                        y_shift_filt = y_shift_ist
            print y_shift_filt
            y_shift_weight = [elt*count for elt,count in y_shift_filt]
            y_freq_list = [count for elt,count in y_shift_filt]
            y_divider = np.sum(y_freq_list)
            x_shift = float(np.sum(x_shift_weight)) / float(x_divider)
            y_shift = float(np.sum(y_shift_weight)) / float(y_divider)

            print 'SURF shift -> X: ' + str(x_shift) + ' Y: ' + str(y_shift)
    
            for image_tg in target_files:
                shutil.copyfile(image_tg,image_tg[:-4]+'_adj_surf.tif')
                rows_target,cols_target,nbands_target,geotransform_target,projection = read_image_parameters(image_tg)
                new_lon = float(x_shift*geotransform_target[1]+geotransform_target[0]) 
                new_lat = float(geotransform_target[3]+y_shift*geotransform_target[5])
                fixed_geotransform = [new_lon,geotransform_target[1],0.0,new_lat,0.0,geotransform_target[5]]
    
                up_image = osgeo.gdal.Open(image_tg[:-4]+'_adj_surf.tif', GA_Update)
                up_image.SetGeoTransform(fixed_geotransform)
                up_image = None
            
            if compute_ndvi == True:      
                print 'Computing NDVI...'    
                ccd_ndvi_files = os.listdir(ccd_folder)
                #print ccd_ndvi_files
                #band3_file = [str(ccd_folder) + '\\' + str(c) for c in ccd_ndvi_files if (".tif" in c or ".TIF" in c) and "xml" not in c and "CCD" in c and str(path_row_list_ccd[f][0]) in c and str(path_row_list_ccd[f][1]) in c and "BAND3_adj" in c]
                #print band3_file
                #band4_file = [str(ccd_folder) + '\\' + str(c) for c in ccd_ndvi_files if (".tif" in c or ".TIF" in c) and "xml" not in c and "CCD" in c and str(path_row_list_ccd[f][0]) in c and str(path_row_list_ccd[f][1]) in c and "BAND4_adj" in c]
                band3_list = read_image(band3_file[0],np.uint8,0)
                band4_list = read_image(band4_file[0],np.uint8,0)
                rows_ndvi,cols_ndvi,nbands_ndvi,geot_ndvi,proj_ndvi = read_image_parameters(band3_file[0])
                ndvi = (band4_list[0].astype(float)-band3_list[0].astype(float)) / (band4_list[0].astype(float)+band3_list[0].astype(float))
                write_image([ndvi],np.float32,0,ccd_folder+'\\ndvi_'+str(path)+'_'+str(row)+'.tif',rows_ndvi,cols_ndvi,fixed_geotransform,proj_ndvi)
            
            os.remove(ccd_folder+'\\multi_class.tif')
            os.remove(ccd_folder+'\\multi.tif')
            
        else:
            print 'Missing HRC correspondence...' + str(path_row_list_ccd[f][0]) + ', ' + str(path_row_list_ccd[f][1])
    end_time = time.time()
    print 'Total time: ' + str(end_time-start_time)

if __name__ == '__main__':
    main()