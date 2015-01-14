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
from operator import itemgetter, attrgetter
from numpy.fft import fft2, ifft2, fftshift


def data_type2gdal_data_type(data_type):
    
    '''Conversion from numpy data type to GDAL data type
    
    :param data_type: numpy type (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type).
    :returns: corresponding GDAL data type
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    ''' 
    #Function needed when it is necessary to write an output file
    if data_type == np.uint16:
        return GDT_UInt16
    if data_type == np.uint8:
        return GDT_Byte
    if data_type == np.int32:
        return GDT_Int32
    if data_type == np.float32:
        return GDT_Float32
    if data_type == np.float64:
        return GDT_Float64
    
    
def read_image(input_raster,data_type,band_selection):
    
    '''Read raster using GDAL
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string).
    :param data_type: numpy type used to read the image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type).
    :param band_selection: number associated with the band to extract (0: all bands, 1: blue, 2: green, 3:red, 4:infrared) (integer).
    :returns:  a list containing the desired bands as ndarrays (list of arrays).
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    ''' 

    band_list = []
    
    if data_type == 0: #most of the images (MR and HR) can be read as uint16
        data_type = np.uint16
        
    inputimg = osgeo.gdal.Open(input_raster, GA_ReadOnly)
    cols=inputimg.RasterXSize
    rows=inputimg.RasterYSize
    nbands=inputimg.RasterCount
    
    if band_selection == 0:
        #read all the bands
        for i in range(1,nbands+1):
            inband = inputimg.GetRasterBand(i) 
            #mat_data = inband.ReadAsArray(0,0,cols,rows).astype(data_type)
            mat_data = inband.ReadAsArray().astype(data_type)
            band_list.append(mat_data) 
    else:
        #read the single band
        inband = inputimg.GetRasterBand(band_selection) 
        mat_data = inband.ReadAsArray(0,0,cols,rows).astype(data_type)
        band_list.append(mat_data)
    
    inputimg = None    
    return band_list


def read_image_parameters(input_raster):
    
    '''Read raster parameters using GDAL
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string).
    :returns:  a list containing rows, columns, number of bands, geo-transformation matrix and projection.
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    ''' 
   
    inputimg = osgeo.gdal.Open(input_raster, GA_ReadOnly)
    cols=inputimg.RasterXSize
    rows=inputimg.RasterYSize
    nbands=inputimg.RasterCount
    geo_transform = inputimg.GetGeoTransform()
    projection = inputimg.GetProjection()
    
    inputimg = None
    return rows,cols,nbands,geo_transform,projection


def world2pixel(geo_transform, long, lat):
    
    '''Conversion from geographic coordinates to matrix-related indexes
    
    :param geo_transform: geo-transformation matrix containing coordinates and resolution of the output (array of 6 elements, float)
    :param long: longitude of the desired point (float)
    :param lat: latitude of the desired point (float)
    :returns: A list with matrix-related x and y indexes (x,y)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    '''
    
    ulX = geo_transform[0] #starting longitude
    ulY = geo_transform[3] #starting latitude
    xDist = geo_transform[1] #x resolution
    yDist = geo_transform[5] #y resolution

    pixel_x = int((long - ulX) / xDist)
    pixel_y = int((ulY - lat) / abs(yDist))
    return (pixel_x, pixel_y)


def write_image(band_list,data_type,band_selection,output_raster,rows,cols,geo_transform,projection):
   
    '''Write array to file as raster using GDAL
    
    :param band_list: list of arrays containing the different bands to write (list of arrays).
    :param data_type: numpy data type of the output image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type)
    :param band_selection: number associated with the band to write (0: all, 1: blue, 2: green, 3: red, 4: infrared) (integer)
    :param output_raster: path and name of the output raster to create (*.TIF, *.tiff) (string)
    :param rows: rows of the output raster (integer)
    :param cols: columns of the output raster (integer)
    :param geo_transform: geo-transformation matrix containing coordinates and resolution of the output (array of 6 elements, float)
    :param projection: projection of the output image (string)
    :returns: An output file is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    '''

    if data_type == 0:
        gdal_data_type = GDT_UInt16 #default data type
    else:
        gdal_data_type = data_type2gdal_data_type(data_type)
    
    driver = osgeo.gdal.GetDriverByName('GTiff')

    if band_selection == 0:
        nbands = len(band_list)
    else:
        nbands = 1
    outDs = driver.Create(output_raster, cols, rows,nbands, gdal_data_type)
    if outDs is None:
        print 'Could not create output file'
        sys.exit(1)
        
    if band_selection == 0:
        #write all the bands to file
        for i in range(0,nbands): 
            outBand = outDs.GetRasterBand(i+1)
            outBand.WriteArray(band_list[i], 0, 0)
    else:
        #write the specified band to file
        outBand = outDs.GetRasterBand(1)   
        outBand.WriteArray(band_list[band_selection-1], 0, 0)
    #assign geomatrix and projection
    outDs.SetGeoTransform(geo_transform)
    outDs.SetProjection(projection)
    outDs = None


def clip_rectangular(input_raster,data_type,input_shape,output_raster,option="standard_roi"):
    
    '''Clip a raster with a rectangular shape based on the provided polygon
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param data_type: numpy type used to read the image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type)
    :param input_shape: path and name of the input shapefile (*.shp) (string)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :returns:  an output file is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    ''' 

    x_list = []
    y_list = []
    # get the shapefile driver
    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    # open the data source
    datasource = driver.Open(input_shape, 0)
    if datasource is None:
        print 'Could not open shapefile'
        sys.exit(1)

    layer = datasource.GetLayer() #get the shapefile layer
    
    inb = osgeo.gdal.Open(input_raster, GA_ReadOnly)
    if inb is None:
        print 'Could not open'
        sys.exit(1)
        
    geoMatrix = inb.GetGeoTransform()
    driver = inb.GetDriver()
    cols = inb.RasterXSize
    rows = inb.RasterYSize
    nbands = inb.RasterCount  
    # loop through the features in the layer
    feature = layer.GetNextFeature()
    while feature:
        # get the x,y coordinates for the point
        geom = feature.GetGeometryRef()
        ring = geom.GetGeometryRef(0)
        n_vertex = ring.GetPointCount()
        for i in range(0,n_vertex-1):
            lon,lat,z = ring.GetPoint(i)
            x_matrix,y_matrix = world2pixel(geoMatrix,lon,lat)
            x_list.append(x_matrix)
            y_list.append(y_matrix)
        # destroy the feature and get a new one
        feature.Destroy()
        feature = layer.GetNextFeature()
    #regularize the shape
    x_list.sort()
    x_min = x_list[0]
    y_list.sort()
    y_min = y_list[0]
    x_list.sort(None, None, True)
    x_max = x_list[0]
    y_list.sort(None, None, True)
    y_max = y_list[0]
    lon_min = float(x_min*geoMatrix[1]+geoMatrix[0]) 
    lat_min = float(geoMatrix[3]+y_min*geoMatrix[5])

    #if x_min < geoMatrix[0]: x_min = geoMatrix[0]
    #if y_min < geoMatrix[3]-rows*geoMatrix[1]: y_min = geoMatrix[3]-rows*geoMatrix[1]
    #if x_max > geoMatrix[0]+cols*geoMatrix[1]: x_max = geoMatrix[0]+cols*geoMatrix[1]
    #if y_max > geoMatrix[3]: y_max = geoMatrix[3]

    #lon_min = x_min
    #lat_min = y_max
    
    #compute the new starting coordinates
    #x_min, y_max = world2pixel(geoMatrix, x_min, y_max)
    #x_max, y_min = world2pixel(geoMatrix, x_max, y_min)

    geotransform = [lon_min,geoMatrix[1],0.0,lat_min,0.0,geoMatrix[5]]
    cols_out = x_max-x_min
    rows_out = y_max-y_min
    '''
    print 'x_max: ' + str(x_max)
    print 'y_max: ' + str(y_max)
    print 'x_min: ' + str(x_min)
    print 'y_min: ' + str(y_min)
    '''
    if option == 'from_class':
        if cols_out > 1000: cols_out = 1000
        if rows_out > 1000: rows_out = 1000
        
    cols_ext = cols_out
    rows_ext = rows_out
    #Fix dimensions
    if x_max > cols: cols_ext = cols - x_min
    if y_max > rows: rows_ext = rows - y_min
    
    if rows_ext > rows_out: rows_ext = rows_out
    if cols_ext > cols_out: cols_ext = cols_out
    
    gdal_data_type = data_type2gdal_data_type(data_type)
    output=driver.Create(output_raster,cols_out,rows_out,nbands,gdal_data_type) #to check
    #print 'cols,rows out'
    #print cols_out,rows_out
    #print 'cols,rows ext'
    #print cols_ext,rows_ext
    for b in range (1,nbands+1):
        inband = inb.GetRasterBand(b)
        data = inband.ReadAsArray(x_min,y_min,cols_ext,rows_ext).astype(data_type)
        if cols_out > cols_ext:
            diff = cols_out - cols_ext
            cols_ext = cols_out
            data = np.hstack((data,np.zeros((rows_ext,diff)))).astype(data_type)
        if rows_out > rows_ext:
            diff = rows_out - rows_ext
            rows_ext = rows_out
            data = np.vstack((data,np.zeros((diff,cols_ext)))).astype(data_type)
        outband = output.GetRasterBand(b)
        outband.WriteArray(data,0,0) #write to output image
    
    output.SetGeoTransform(geotransform) #set the transformation
    output.SetProjection(inb.GetProjection())
    # close the data source and text file
    datasource.Destroy()


def EUC_SURF(ref_band_mat,target_band_mat,output_as_array):
     
    '''
    SURF version used for Landsat by EUCENTRE
    
    :param ref_band_mat: numpy 8 bit array containing reference image
    :param target_band_mat: numpy 8 bit array containing target image
    :param output_as_array: if True the output is converted to matrix for visualization purposes
    :returns: points from reference, points from target, result of matching function or array of points (depending on the output_as_array flag)
    
    '''
    
    detector = cv2.FeatureDetector_create("SURF") #Detector definition
    descriptor = cv2.DescriptorExtractor_create("BRIEF") #Descriptor definition
    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming") #Matcher definition
    
    #Extraction of features from REFERENCE
    ref_mask_zeros = np.ma.masked_equal(ref_band_mat, 0).astype('uint8')
    k_ref = detector.detect(ref_band_mat.astype('uint8'), mask=ref_mask_zeros)
    kp_ref, d_ref = descriptor.compute(ref_band_mat, k_ref)
    h_ref, w_ref = ref_band_mat.shape[:2]
    ref_band_mat = []

    #Extration of features from TARGET
    target_mask_zeros = np.ma.masked_equal(target_band_mat, 0).astype('uint8')
    k_target = detector.detect(target_band_mat.astype('uint8'), mask=target_mask_zeros)
    kp_target, d_target = descriptor.compute(target_band_mat, k_target)
    h_target, w_target = target_band_mat.shape[:2]
    target_band_mat = []

    #Matching
    matches = matcher.match(d_ref, d_target)
    matches = sorted(matches, key = lambda x:x.distance)
    matches_disp = matches[:3]
    #matches_disp = matches
    if output_as_array == True:
        ext_points = np.zeros(shape=(len(matches_disp),4))
        i = 0
        for m in matches_disp:
            ext_points[i][:]= [int(kp_ref[m.queryIdx].pt[0]),int(kp_ref[m.queryIdx].pt[1]),int(kp_target[m.trainIdx].pt[0]),int(kp_target[m.trainIdx].pt[1])]
            i = i+1
        return kp_ref,kp_target,ext_points
    else:
        return kp_ref,kp_target,matches


def ORIGINAL_SURF(ref_band_mat,target_band_mat,output_as_array):

    '''
    SURF version taken from website example
    http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html#surf
    https://gist.github.com/moshekaplan/5106221

    :param ref_band_mat: numpy 8 bit array containing reference image
    :param target_band_mat: numpy 8 bit array containing target image
    :param output_as_array: if True the output is converted to matrix for visualization purposes
    :returns: points from reference, points from target, result of matching function or array of points (depending on the output_as_array flag)

    '''

    detector = cv2.SURF(400) #Detector definition, Hessian threshold set to 400 (suggested between 300 and 500)
    detector.extended = True #Descriptor extended to 128
    detector.upright = True #Avoid orientation
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    #Extraction of features from REFERENCE
    ref_mask_zeros = np.ma.masked_equal(ref_band_mat, 0).astype('uint8')
    kp_ref, des_ref = detector.detectAndCompute(ref_band_mat, mask=ref_mask_zeros)
    h_ref, w_ref = ref_band_mat.shape[:2]
    print h_ref, w_ref
    ref_band_mat = []

    #Extraction of features from TARGET
    target_mask_zeros = np.ma.masked_equal(target_band_mat, 0).astype('uint8')
    kp_target, des_target = detector.detectAndCompute(target_band_mat, mask=target_mask_zeros)
    h_target, w_target = target_band_mat.shape[:2]
    print h_target, w_target
    target_band_mat = []

    #Matching
    matches = matcher.match(des_ref,des_target)

    if output_as_array == True:
        ext_points = np.zeros(shape=(len(matches),4))
        i = 0
        for m in matches:
            ext_points[i][:]= [int(kp_ref[m.queryIdx].pt[0]),int(kp_ref[m.queryIdx].pt[1]),int(kp_target[m.trainIdx].pt[0]),int(kp_target[m.trainIdx].pt[1])]
            i = i+1
        return kp_ref,kp_target,ext_points
    else:
        return kp_ref,kp_target,matches

    
def EUC_filter(kp_ref,kp_target,matches, thres_dist = 100):

    '''
    Filter for matching points originally developed for Landsat by EUCENTRE

    :param kp_ref: points extracted from reference
    :param kp_target: points extracted from target
    :param matches: output of the matching function
    :param thres_dist: limit value for Hamming distance (default value = 100)
    :returns: best extracted point

    '''

    if thres_dist == 0:
        thres_dist = 100
    sel_matches = [m for m in matches if m.distance <= thres_dist]

    #Define structures
    ext_points = np.zeros(shape=(len(sel_matches),4))
    ext_points_shift = np.zeros(shape=(len(sel_matches),2))
    ext_points_shift_abs = np.zeros(shape=(len(sel_matches),1))
    i = 0
    compar_stack = np.array([100,1.5,0.0,1,1,2,2])
    for m in sel_matches:
        ext_points[i][:]= [int(kp_ref[m.queryIdx].pt[0]),int(kp_ref[m.queryIdx].pt[1]),int(kp_target[m.trainIdx].pt[0]),int(kp_target[m.trainIdx].pt[1])]
        ext_points_shift[i][:] = [int(kp_target[m.trainIdx].pt[0])-int(kp_ref[m.queryIdx].pt[0]),int(kp_target[m.trainIdx].pt[1])-int(kp_ref[m.queryIdx].pt[1])]
        ext_points_shift_abs [i][:] = [np.sqrt((int(cols_ref + kp_target[m.trainIdx].pt[0])-int(kp_ref[m.queryIdx].pt[0]))**2+
                                           (int(kp_target[m.trainIdx].pt[1])-int(kp_ref[m.queryIdx].pt[1]))**2)]
        
        deltax = np.float(int(kp_target[m.trainIdx].pt[0])-int(kp_ref[m.queryIdx].pt[0]))
        deltay = np.float(int(kp_target[m.trainIdx].pt[1])-int(kp_ref[m.queryIdx].pt[1]))
        
        if deltax == 0 and deltay != 0:
            slope = 90
        elif deltax == 0 and deltay == 0:
            slope = 0
        else:
            slope = (np.arctan(deltay/deltax)*360)/(2*np.pi)
        
        compar_stack = np.vstack([compar_stack,[m.distance,ext_points_shift_abs [i][:],slope,int(kp_ref[m.queryIdx].pt[0]),int(kp_ref[m.queryIdx].pt[1]),int(kp_target[m.trainIdx].pt[0]),int(kp_target[m.trainIdx].pt[1])]])
        i=i+1

    #Filtering 
    compar_stack = compar_stack[compar_stack[:,0].argsort()] #Returns the indices that would sort an array.
    best = select_best_matching_EUC(compar_stack[0:90]) #The number of sorted points to be passed


    best_match = [best[3:7]]
    return best_match


def select_best_matching_EUC(compstack):

    '''
    Determine the best matching points among the extracted ones

    :param compstack: array with points and distances extracted by the gcp_extraction function
    :returs: index of the row with the best matching points
    
    Author: Mostapha Harb - Daniele De Vecchi - Daniel Aurelio Galeazzo
    Last modified: 23/05/2014
    '''
    
    # Sort
    compstack = compstack[compstack[:,2].argsort()]
    spl_slope = np.append(np.where(np.diff(compstack[:,2])>0.1)[0]+1,len(compstack[:,0]))
    
    step = 0
    best_variability = 5
    len_bestvariab = 0
    best_row = np.array([100,1.5,0.0,1,1,2,2])

    for i in spl_slope:
        slope = compstack[step:i][:,2]
        temp = compstack[step:i][:,1]
        variab_temp = np.var(temp)
        count_list=[]
        if variab_temp <= best_variability and len(temp) >3:
            count_list.append(len(temp))

            if variab_temp < best_variability:
                
                best_variability = variab_temp
                len_bestvariab = len(temp)                
                best_row = compstack[step:i][compstack[step:i][:,0].argsort()][0]
                all_rows = compstack[step:i]
            if variab_temp == best_variability:
                if len(temp)>len_bestvariab:
                    best_variability = variab_temp
                    len_bestvariab = len(temp)                
                    best_row = compstack[step:i][compstack[step:i][:,0].argsort()][0]
                    all_rows = compstack[step:i]
        step = i
    return best_row #,,point_list1,point_list2


def ORIGINAL_filter(kp_ref,kp_target,matches,ratio = 0.75):

    '''
    Original filter from website example
    https://gist.github.com/moshekaplan/5106221

    :param matches: output of the matching operation
    :returns: list of extracted points

    '''

    sel_matches = [m for m in matches if m[0].distance < m[1].distance * ratio]
    ext_points = np.zeros(shape=(len(sel_matches),4))
    i = 0
    for m in matches:
        ext_points[i][:]= [int(kp_ref[m.queryIdx].pt[0]),int(kp_ref[m.queryIdx].pt[1]),int(kp_target[m.trainIdx].pt[0]),int(kp_target[m.trainIdx].pt[1])]
        i = i+1

    return ext_points
    

def window_output(window_name,ext_points,ref_band_mat,target_band_mat):

    '''
    Define a window to show the extracted points, useful to refine the method

    :param window_name: name of the output window
    :param ext_points: array with coordinates of extracted points
    :param ref_band_mat: numpy 8 bit array containing reference image
    :param target_band_mat: numpy 8 bit array containing target image
    :returns: a window is shown with the matching points

    '''
    #Height (rows) and width (cols) of the reference image
    h_ref, w_ref = ref_band_mat.shape[:2]
    #Height (rows) and width (cols) of the target image
    h_target, w_target = target_band_mat.shape[:2]
    
    vis = np.zeros((max(h_ref, h_target), w_ref+w_target), np.uint8)
    vis[:h_ref, :w_ref] = ref_band_mat
    vis[:h_target, w_ref:w_ref+w_target] = target_band_mat
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    green = (0, 255, 0) #green
    print 'Total matching: ' + str(len(ext_points))
    for p in range(0,len(ext_points)):
        x_ref,y_ref,x_target,y_target = int(ext_points[p][0]),int(ext_points[p][1]),int(ext_points[p][2]),int(ext_points[p][3])
        #print x_ref,y_ref,x_target,y_target
        print x_ref-x_target,y_ref-y_target
        cv2.circle(vis, (x_ref, y_ref), 2, green, -1)
        cv2.circle(vis, (x_target+w_ref, y_target), 2, green, -1)
        cv2.line(vis, (x_ref, y_ref), (x_target+w_ref, y_target), green)
    cv2.imwrite("F:\\PhD\\Rio_de_Janeiro\\CBERS_data\\San_Paolo\\extract_screenshots\\selected_matching.png", vis)
    vis0 = vis.copy()
    cv2.imshow(window_name, vis)
    cv2.waitKey()
    cv2.destroyAllWindows() 


def FFT_coregistration(ref_band_mat,target_band_mat):

    '''
    Alternative method used to coregister the images based on the FFT

    :param ref_band_mat: numpy 8 bit array containing reference image
    :param target_band_mat: numpy 8 bit array containing target image
    :returns: the shift among the two input images 

    '''

    #Normalization - http://en.wikipedia.org/wiki/Cross-correlation#Normalized_cross-correlation 
    ref_band_mat = (ref_band_mat - ref_band_mat.mean()) / ref_band_mat.std()
    target_band_mat = (target_band_mat - target_band_mat.mean()) / target_band_mat.std() 

    #Check dimensions - they have to match
    rows_ref,cols_ref =  ref_band_mat.shape
    rows_target,cols_target = target_band_mat.shape

    if rows_target < rows_ref:
        print 'Rows - correction needed'

        diff = rows_ref - rows_target
        target_band_mat = np.vstack((target_band_mat,np.zeros((diff,cols_target))))
    elif rows_ref < rows_target:
        print 'Rows - correction needed'
        diff = rows_target - rows_ref
        ref_band_mat = np.vstack((ref_band_mat,np.zeros((diff,cols_ref))))
        
    rows_target,cols_target = target_band_mat.shape
    rows_ref,cols_ref = ref_band_mat.shape

    if cols_target < cols_ref:
        print 'Columns - correction needed'
        diff = cols_ref - cols_target
        target_band_mat = np.hstack((target_band_mat,np.zeros((rows_target,diff))))
    elif cols_ref < cols_target:
        print 'Columns - correction needed'
        diff = cols_target - cols_ref
        ref_band_mat = np.hstack((ref_band_mat,np.zeros((rows_ref,diff))))

    rows_target,cols_target = target_band_mat.shape   

    #translation(im_target,im_ref)
    freq_target = fft2(target_band_mat)   
    freq_ref = fft2(ref_band_mat)  
    inverse = abs(ifft2((freq_target * freq_ref.conjugate()) / (abs(freq_target) * abs(freq_ref))))   

    #Converts a flat index or array of flat indices into a tuple of coordinate arrays. would give the pixel of the max inverse value
    y_shift,x_shift = np.unravel_index(np.argmax(inverse),(rows_target,cols_target))

    if y_shift > rows_target // 2: # // used to truncate the division
        y_shift -= rows_target
    if x_shift > cols_target // 2: # // used to truncate the division
        x_shift -= cols_target
    
    return -x_shift, -y_shift


def resampling(input_raster,output_raster,output_resolution,resampling_algorithm):
    
    '''
    Resampling operation using OTB library
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :param output_resolution: resolution of the outout raster file (float)
    :param resampling_algorithm: choice among different algorithms (nearest_neigh,linear,bicubic)
    :returns:  an output file is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    '''
    
    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_raster)
    scale_value = round(float(geo_transform[1])/float(output_resolution),4)
    if resampling_algorithm == 'nearest_neigh': 
        interp = 'nn'
    if resampling_algorithm == 'linear':
        interp = 'linear'
    if resampling_algorithm == 'bicubic':
        interp = 'bco'
    command = 'C:/OSGeo4W64/bin/otbcli_RigidTransformResample -progress 1 -in {} -out {} uint8 -transform.type id -transform.type.id.scalex {} -transform.type.id.scaley {} -interpolator {}'.format(input_raster,output_raster,scale_value,scale_value,interp)
    proc = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,).stdout
    print 'Resampling: ' + str(input_raster)
    for line in iter(proc.readline, ''): 
        if '[*' in line:
            idx = line.find('[*')
            perc = int(line[idx - 4:idx - 2].strip(' '))
            if perc%10 == 0 and perc!=0:
                print str(perc) + '...',
            
    print '100'
    '''
    RigidTransformResample = otbApplication.Registry.CreateApplication("RigidTransformResample") 
    # The following lines set all the application parameters: 
    RigidTransformResample.SetParameterString("in", input_raster) 
    RigidTransformResample.SetParameterString("out", output_raster) 
    RigidTransformResample.SetParameterString("transform.type","id") 
    RigidTransformResample.SetParameterFloat("transform.type.id.scalex", scale_value) 
    RigidTransformResample.SetParameterFloat("transform.type.id.scaley", scale_value) 
    
    if resampling_algorithm == 'nearest_neigh': 
        RigidTransformResample.SetParameterString("interpolator","nn")
    if resampling_algorithm == 'linear':
        RigidTransformResample.SetParameterString("interpolator","linear")
    if resampling_algorithm == 'bicubic':
        RigidTransformResample.SetParameterString("interpolator","bco")

    RigidTransformResample.ExecuteAndWriteOutput()
    '''


def get_coordinate_limit(input_raster):

    '''Get corner cordinate from a raster

    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :returs: minx,miny,maxx,maxy: points taken from geomatrix (string)
    
    Author: Daniel Aurelio Galeazzo - Daniele De Vecchi - Mostapha Harb
    Last modified: 23/05/2014
    '''
    dataset = osgeo.gdal.Open(input_raster, GA_ReadOnly)
    if dataset is None:
        print 'Could not open'
        sys.exit(1)
    driver = dataset.GetDriver()
    band = dataset.GetRasterBand(1)

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    geoMatrix = dataset.GetGeoTransform()
    minx = geoMatrix[0]
    miny = geoMatrix[3] + width*geoMatrix[4] + height*geoMatrix[5] 
    maxx = geoMatrix[0] + width*geoMatrix[1] + height*geoMatrix[2]
    maxy = geoMatrix[3]

    dataset = None

    return minx,miny,maxx,maxy


def extract_tiles(input_raster,start_col_coord,start_row_coord,end_col_coord,end_row_coord,data_type):
    
    '''
    Extract a subset of a raster according to the desired coordinates

    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :param start_col_coord: starting longitude coordinate
    :param start_row_coord: starting latitude coordinate
    :param end_col_coord: ending longitude coordinate
    :param end_row_coord: ending latitude coordinate

    :returns: an output file is created and also a level of confidence on the tile is returned

    Author: Daniele De Vecchi
    Last modified: 20/08/2014
    '''

    #Read input image
    rows,cols,nbands,geotransform,projection = read_image_parameters(input_raster)
    band_list = read_image(input_raster,data_type,0)
    #Definition of the indices used to tile
    start_col_ind,start_row_ind = world2pixel(geotransform,start_col_coord,start_row_coord)
    end_col_ind,end_row_ind = world2pixel(geotransform,end_col_coord,end_row_coord)
    #print start_col_ind,start_row_ind
    #print end_col_ind,end_row_ind
    #New geotransform matrix
    new_geotransform = [start_col_coord,geotransform[1],0.0,start_row_coord,0.0,geotransform[5]]
    #Extraction
    data = band_list[0][start_row_ind:end_row_ind,start_col_ind:end_col_ind]
    
    band_list = []
    return data,start_col_coord,start_row_coord,end_col_coord,end_row_coord


def tile_statistics(band_mat,start_col_coord,start_row_coord,end_col_coord,end_row_coord):

    '''
    Compute statistics related to the input tile

    :param band_mat: numpy 8 bit array containing the extracted tile
    :param start_col_coord: starting longitude coordinate
    :param start_row_coord: starting latitude coordinate
    :param end_col_coord: ending longitude coordinate
    :param end_row_coord: ending latitude coordinate

    :returns: a list of statistics (start_col_coord,start_row_coord,end_col_coord,end_row_coord,confidence, min frequency value, max frequency value, standard deviation value, distance among frequent values)

    Author: Daniele De Vecchi
    Last modified: 22/08/2014
    '''

    #Histogram definition
    data_flat = band_mat.flatten()
    data_counter = collections.Counter(data_flat)
    data_common = (data_counter.most_common(20)) #20 most common values
    data_common_sorted = sorted(data_common,key=itemgetter(0)) #reverse=True for inverse order
    hist_value = [elt for elt,count in data_common_sorted]
    hist_count = [count for elt,count in data_common_sorted]

    #Define the level of confidence according to the computed statistics 
    min_value = hist_value[0]
    max_value = hist_value[-1]
    std_value = np.std(hist_count)
    diff_value = max_value - min_value
    min_value_count = hist_count[0]
    max_value_count = hist_count[-1] 
    tot_count = np.sum(hist_count)
    min_value_freq = (float(min_value_count) / float(tot_count)) * 100
    max_value_freq = (float(max_value_count) / float(tot_count)) * 100

    if max_value_freq > 20.0 or min_value_freq > 20.0 or diff_value < 18 or std_value > 100000:
        confidence = 0
    elif max_value_freq > 5.0: #or std_value < 5.5: #or min_value_freq > 5.0:
        confidence = 0.5
    else:
        confidence = 1

    return (start_col_coord,start_row_coord,end_col_coord,end_row_coord,confidence,min_value_freq,max_value_freq,std_value,diff_value)


def slope_filter(ext_points):

    '''
    Filter based on the deviation of the slope

    :param ext_points: array with coordinates of extracted points
    :returns: an array of filtered points

    Author: Daniele De Vecchi
    Last modified: 19/08/2014
    '''
    
    discard_list = []
    for p in range(0,len(ext_points)):
        #The first point is the one with minimum distance so it is supposed to be for sure correct
        x_ref,y_ref,x_target,y_target = int(ext_points[p][0]),int(ext_points[p][1]),int(ext_points[p][2]),int(ext_points[p][3])
        if x_target-x_ref != 0:
            istant_slope = float((y_target-y_ref)) / float((x_target-x_ref))
        else:
            istant_slope = 0
        if p == 0:
            slope_mean = istant_slope
        else:
            slope_mean = float(slope_mean+istant_slope) / float(2)
        slope_std = istant_slope - slope_mean
        if abs(slope_std) >= 0.1:
            discard_list.append(p)
        #print 'istant_slope: ' + str(istant_slope)
        #print 'slope_mean: ' + str(slope_mean)
        #print 'slope_std: ' + str(slope_std)
        
    new_points = np.zeros(shape=(len(ext_points)-len(discard_list),4))
    #print discard_list
    p = 0
    for dp in range(0,len(ext_points)):
        if dp not in discard_list:
            new_points[p][:]= int(ext_points[dp][0]),int(ext_points[dp][1]),int(ext_points[dp][2]),int(ext_points[dp][3])
            p = p+1
        else:
            dp = dp+1
    return new_points


def unsupervised_classification_otb(input_raster,output_raster,n_classes,n_iterations):
    
    '''Unsupervised K-Means classification using OTB library.
    Tool used to recall the K-Means unsupervised classification algorithm implemented by the Orfeo Toolbox library. User input is limited to the number of classes to extract and the number
    of iterations of the classifier.

    Example: unsupervised_classification_otb(input_raster='C:\\Users\\Guest\\work\\pansharp.tif',output_raster='C:\\Users\\Guest\\work\\pansharp_unsup.tif',n_classes=5,n_iterations=10)

    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :param n_classes: number of classes to extract (integer)
    :param n_iterations: number of iterations of the classifier (integer)
    :returns:  an output raster is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 20/03/2014
    '''
    
    KMeansClassification = otbApplication.Registry.CreateApplication("KMeansClassification") 
 
    # The following lines set all the application parameters: 
    KMeansClassification.SetParameterString("in", input_raster) 
    KMeansClassification.SetParameterInt("ts", 1000) 
    KMeansClassification.SetParameterInt("nc", n_classes) 
    KMeansClassification.SetParameterInt("maxit", n_iterations) 
    KMeansClassification.SetParameterFloat("ct", 0.0001) 
    KMeansClassification.SetParameterString("out", output_raster) 
    
    # The following line execute the application 
    KMeansClassification.ExecuteAndWriteOutput()


def classification_statistics(input_raster_classification,input_raster):

    '''
    Compute statistics related to the input unsupervised classification

    :param input_raster_classification: path and name of the input raster file with classification(*.TIF,*.tiff) (string)
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)

    :returns: a list of statistics (value/class,min_value,max_value,diff_value,std_value,min_value_freq,max_value_freq,tot_count)

    Author: Daniele De Vecchi
    Last modified: 25/08/2014
    '''

    band_list_classification = read_image(input_raster_classification,np.uint8,0)
    rows_class,cols_class,nbands_class,geotransform_class,projection_class = read_image_parameters(input_raster_classification)

    band_list = read_image(input_raster,np.uint8,0)
    rows,cols,nbands,geotransform,projection = read_image_parameters(input_raster)

    max_class = np.max(band_list_classification[0])
    stat_list = []
    for value in range(0,max_class+1):
        #print '----------------------------'
        #print 'Class ' + str(value)
        mask = np.equal(band_list_classification[0],value)
        data = np.extract(mask,band_list[0])

        #Statistics
        #Histogram definition
        data_flat = data.flatten()
        data_counter = collections.Counter(data_flat)
        data_common = (data_counter.most_common(20)) #20 most common values
        data_common_sorted = sorted(data_common,key=itemgetter(0)) #reverse=True for inverse order
        hist_value = [elt for elt,count in data_common_sorted]
        hist_count = [count for elt,count in data_common_sorted]

        #Define the level of confidence according to the computed statistics 
        min_value = hist_value[0]
        max_value = hist_value[-1]
        std_value = np.std(hist_count)
        diff_value = max_value - min_value
        min_value_count = hist_count[0]
        max_value_count = hist_count[-1] 
        tot_count = np.sum(hist_count)
        min_value_freq = (float(min_value_count) / float(tot_count)) * 100
        max_value_freq = (float(max_value_count) / float(tot_count)) * 100

        #print 'Min value: ' + str(min_value)
        #print 'Max value: ' + str(max_value)
        #print 'Diff value: ' + str(diff_value)
        #print 'Standard Deviation: ' + str(std_value)
        #print 'Min value frequency: ' + str(min_value_freq)
        #print 'Max value frequency: ' + str(max_value_freq)
        #print 'Total values: ' + str(tot_count)
        #print '----------------------------'
        stat_list.append((value,min_value,max_value,diff_value,std_value,min_value_freq,max_value_freq,tot_count))
    return stat_list



def layer_stack(input_raster_list,output_raster,data_type):
    
    '''Merge single-band files into one multi-band file
    
    :param input_raster_list: list with paths and names of the input raster files (*.TIF,*.tiff) (list of strings)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :param data_type: numpy type used to read the image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type)
    :returns:  an output file is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    ''' 
    
    final_list = []
    for f in range(0,len(input_raster_list)): #read image by image
        band_list = read_image(input_raster_list[f],data_type,0)
        rows,cols,nbands,geo_transform,projection = read_image_parameters(input_raster_list[f])
        final_list.append(band_list[0]) #append every band to a unique list
        
    write_image(final_list,data_type,0,output_raster,rows,cols,geo_transform,projection) #write the list to output file


def rast2shp(input_raster,output_shape,mask_raster = ''):
    
    '''Conversion from raster to shapefile using GDAL
    
    :param input_raster: path and name of the input raster (*.TIF, *.tiff) (string)
    :param output_shape: path and name of the output shapefile to create (*.shp) (string)
    :returns: An output shapefile is created 
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    
    Reference: http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    '''

    src_image = osgeo.gdal.Open(input_raster)
    src_band = src_image.GetRasterBand(1)
    projection = src_image.GetProjection()
    if mask_raster != '':
            mask_image = osgeo.gdal.Open(mask_raster)
            mask_band = mask_image.GetRasterBand(1)
    else:
        mask_band = src_band.GetMaskBand()
    driver_shape=osgeo.ogr.GetDriverByName('ESRI Shapefile')
    outfile=driver_shape.CreateDataSource(output_shape)
    outlayer=outfile.CreateLayer('Conversion',geom_type=osgeo.ogr.wkbPolygon)
    dn = osgeo.ogr.FieldDefn('DN',osgeo.ogr.OFTInteger)
    outlayer.CreateField(dn)
    
    #Polygonize
    osgeo.gdal.Polygonize(src_band,mask_band,outlayer,0)
    
    outprj=osgeo.osr.SpatialReference(projection)
    outprj.MorphToESRI()
    file_prj = open(output_shape[:-4]+'.prj', 'w')
    file_prj.write(outprj.ExportToWkt())
    file_prj.close()
    src_image = None
    outfile = None


def split_shape(input_shape,option="memory",output_shape="out"):
   
    '''Extract a single feature from a shapefile
    
    :param input_layer: layer of a shapefile (shapefile layer)
    :param index: index of the feature to extract (integer)
    :param option: 'memory' or 'file' depending on the desired output (default is memory) (string)
    :param output_shape: path and name of the output shapefile (temporary file) (*.shp) (string)
    :returns:  an output shapefile is created
    :raises: AttributeError, KeyError
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    ''' 

    #TODO: Why do we need this function? Does not seems like a good idea to do this. Why not simply loop through the features?
    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    infile=driver.Open(input_shape,0)
    input_layer=infile.GetLayer()
    '''
    if option == 'file':
        driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    elif option == 'memory':
        driver = osgeo.ogr.GetDriverByName('Memory')
    '''
    layer_defn = input_layer.GetLayerDefn()
    outDS = driver.CreateDataSource(output_shape)
    outlayer = outDS.CreateLayer('polygon', geom_type=osgeo.ogr.wkbPolygon)
    dn_def = osgeo.ogr.FieldDefn('DN', osgeo.ogr.OFTInteger)
    area_def = osgeo.ogr.FieldDefn('Area', osgeo.ogr.OFTReal)
    outlayer.CreateField(dn_def)
    outlayer.CreateField(area_def)
    featureDefn = outlayer.GetLayerDefn()

    # loop through the input features
    infeature = input_layer.GetNextFeature()
    max_area = 0
    feature_count = 0
    
    while infeature:
        geom = infeature.GetGeometryRef()
        area = geom.Area()
        #print area
        dn = infeature.GetField('DN')
        if dn!=0:
            if area > max_area: 
                max_area = area
                selected_feature = feature_count
        infeature = input_layer.GetNextFeature()
        feature_count = feature_count + 1 

    input_layer.ResetReading()
    inFeature = input_layer.GetFeature(selected_feature) 
    outfeature = osgeo.ogr.Feature(featureDefn)
    geom = inFeature.GetGeometryRef()
    area = geom.Area()
    dn = inFeature.GetField('DN')
    outfeature.SetGeometry(geom)
    outfeature.SetField('DN',dn)
    outfeature.SetField('Area',area)
    outlayer.CreateFeature(outfeature)
    outfeature.Destroy()
    infile.Destroy()
    inFeature.Destroy()
    outDS.Destroy()
    shutil.copyfile(input_shape[:-4]+'.prj',output_shape[:-4]+'.prj')


def linear_quantization(input_mat,quantization_factor):
    
    '''Quantization of all the input bands cutting the tails of the distribution
    
    :param input_band_list: list of 2darrays (list of 2darrays)
    :param quantization_factor: number of levels as output (integer)
    :returns:  list of values corresponding to the quantized bands (list of 2darray)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 12/05/2014
    '''

    q_factor = quantization_factor - 1
    inmatrix = input_mat.reshape(-1)
    print np.min(inmatrix),np.max(inmatrix)
    out = np.bincount(inmatrix)
    tot = inmatrix.shape[0]
    freq = (out.astype(np.float32)/float(tot))*100 #frequency for each value
    cumfreqs = np.cumsum(freq)
    first = np.where(cumfreqs>1.49)[0][0] #define occurrence limits for the distribution
    last = np.where(cumfreqs>97.8)[0][0]
    input_mat[np.where(input_mat>last)] = last
    input_mat[np.where(input_mat<first)] = first

    k1 = float(q_factor)/float((last-first)) #k1 term of the quantization formula
    k2 = np.ones(input_mat.shape)-k1*first*np.ones(input_mat.shape) #k2 term of the quantization formula
    out_matrix = np.floor(input_mat*k1+k2) #take the integer part
    out_matrix2 = out_matrix-np.ones(out_matrix.shape)
    out_matrix2.astype(np.uint8)

    return out_matrix2


def create_gdal_gcps(ext_points,geotransform_ref,geotransform_target):
    
    '''
    Function to convert the points extracted from SURF into GCPs compatible with the GDAL transform function

    :param ext_points: array with coordinates of extracted points
    :returns: an array of filtered points

    Author: Daniele De Vecchi
    Last modified: 10/09/2014
    '''
    gdal_gcp = []
    gdal_string = ''
    #ext points structure: x_ref,y_ref,x_target,y_target
    #gdal structure: pixel line easting northing
    for p in range(0,len(ext_points)):
        north_target = geotransform_target[3] + ext_points[p][2]*geotransform_target[4] + ext_points[p][3]*geotransform_target[5] 
        east_target = geotransform_target[0] + ext_points[p][2]*geotransform_target[1] + ext_points[p][3]*geotransform_target[2]
        #pixel_target,line_target =  world2pixel(geotransform_target_original, east_target, north_target)
        pixel_target = east_target
        line_target = north_target
        north_ref = geotransform_ref[3] + ext_points[p][0]*geotransform_ref[4] + ext_points[p][1]*geotransform_ref[5] 
        east_ref = geotransform_ref[0] + ext_points[p][0]*geotransform_ref[1] + ext_points[p][1]*geotransform_ref[2]
        gdal_gcp.append((pixel_target,line_target,east_ref,north_ref))
        gdal_string = gdal_string + '-gcp {} {} {} {} '.format(str(pixel_target),str(line_target),str(east_ref),str(north_ref))
        print gdal_string
    return gdal_gcp


def generate_footprints(input_list,output_shape):

    '''
    Create footprints from rasters and save them to a shapefile

    :param input_list: list of raster files to process
    :param output_shape: output shapefile

    :returns: an output shapefile is created with all footprints and names from files
    
    Author: Daniele De Vecchi
    Last modified: 15/09/2014
    '''

    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    outfile = driver.CreateDataSource(output_shape)
    outlayer=outfile.CreateLayer('Conversion',geom_type=osgeo.ogr.wkbPolygon)
    id_def = osgeo.ogr.FieldDefn('id',osgeo.ogr.OFTInteger)
    name_def = osgeo.ogr.FieldDefn('name',osgeo.ogr.OFTString)
    outlayer.CreateField(id_def)
    outlayer.CreateField(name_def)
    featureDefn = outlayer.GetLayerDefn()
    for c in range(0,len(input_list)):
        #hf = hrc_folder + '\\' + hrc_new_files[c]
        path,file_name = os.path.split(input_list[c])
        band_list = read_image(input_list[c],np.uint8,0)
        rows,cols,nbands,geot,proj = read_image_parameters(input_list[c])
        '''
        mask = np.greater(band_list[0],0)
        out = np.choose(mask,(0,1))
        out = sp.ndimage.binary_closing(out, structure=np.ones((7,7))).astype(np.int)
        '''
        out = np.ones((rows,cols))
        write_image([out],np.uint8,0,input_list[c][:-4]+'_mod.tif',rows,cols,geot,proj)
        rast2shp(input_list[c][:-4]+'_mod.tif',input_list[c][:-4]+'.shp',input_list[c][:-4]+'_mod.tif')
        os.remove(input_list[c][:-4]+'_mod.tif')
        infile = driver.Open(input_list[c][:-4]+'.shp',0)
        input_layer=infile.GetLayer()
        infeature = input_layer.GetNextFeature()
        new_geom = infeature.GetGeometryRef()
        outfeature = osgeo.ogr.Feature(featureDefn)
        outfeature.SetGeometry(new_geom)
        outfeature.SetField('id',c+1)
        outfeature.SetField('name',str(file_name))
        outlayer.CreateFeature(outfeature)
        outfeature.Destroy()
        infile.Destroy()
        if c == 0:
            shutil.copyfile(input_list[c][:-4]+'.prj',output_shape[:-4]+'.prj')  
        os.remove(input_list[c][:-4]+'.prj')
        os.remove(input_list[c][:-4]+'.shp')
        os.remove(input_list[c][:-4]+'.shx')
        os.remove(input_list[c][:-4]+'.dbf')
    outfile.Destroy()


def find_intersecting_tiles(container_shape,tiles_shape):

    '''
    Find all the tiles intersecting the container_shape

    :param container_shape: shapefile used to define the area of interest
    :param tiles_shape: shapefile with footprints of all the tiles of interest

    :returns: a list with names of the intersecting files

    Author: Daniele De Vecchi
    Last modified: 15/09/2014
    '''

    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    infile_ccd = driver.Open(container_shape,0)
    infile_hrc = driver.Open(tiles_shape,0)
    input_layer_ccd=infile_ccd.GetLayer()
    input_layer_hrc=infile_hrc.GetLayer()
    infeature_ccd = input_layer_ccd.GetNextFeature()
    
    ccd_ord_list = []
    while infeature_ccd:
        intersecting_list = []
        geom = infeature_ccd.GetGeometryRef()
        ccd_name = infeature_ccd.GetField('name')
        infeature_hrc = input_layer_hrc.GetNextFeature()
        while infeature_hrc:
            geom_hrc = infeature_hrc.GetGeometryRef()
            area_geom_hrc = geom_hrc.Area()
            intersect = geom_hrc.Intersection(geom)
            if geom_hrc.Intersect(geom):
                area_intersect = intersect.Area()
                perc_area = (float(area_intersect) / float(area_geom_hrc)) *100
                if perc_area > 30.0:
                    intersecting_file = infeature_hrc.GetField('name')
                    intersecting_list.append(intersecting_file)
            infeature_hrc = input_layer_hrc.GetNextFeature()
        ccd_ord_list.append((ccd_name,intersecting_list))
        input_layer_hrc.ResetReading()
        infeature_ccd = input_layer_ccd.GetNextFeature()
       
    infile_ccd.Destroy()
    infile_hrc.Destroy()

    return ccd_ord_list
    #return intersecting_list