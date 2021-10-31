/*******************************************************************************
* Copyright (c) 2020, STMicroelectronics - All Rights Reserved
*
* This file is part of the VL53L5CX Ultra Lite Driver and is dual licensed,
* either 'STMicroelectronics Proprietary license'
* or 'BSD 3-clause "New" or "Revised" License' , at your option.
*
********************************************************************************
*
* 'STMicroelectronics Proprietary license'
*
********************************************************************************
*
* License terms: STMicroelectronics Proprietary in accordance with licensing
* terms at www.st.com/sla0081
*
* STMicroelectronics confidential
* Reproduction and Communication of this document is strictly prohibited unless
* specifically authorized in writing by STMicroelectronics.
*
*
********************************************************************************
*
* Alternatively, the VL53L5CX Ultra Lite Driver may be distributed under the
* terms of 'BSD 3-clause "New" or "Revised" License', in which case the
* following provisions apply instead of the ones mentioned above :
*
********************************************************************************
*
* License terms: BSD 3-clause "New" or "Revised" License.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software
* without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*
*******************************************************************************/

/***********************************/
/*  VL53L5CX ULD motion indicator  */
/***********************************/
/*
* This example shows the VL53L5CX motion indicator capabilities.
* To use this example, user needs to be sure that macro
* VL53L5CX_DISABLE_MOTION_INDICATOR is NOT enabled (see file platform.h).
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "vl53l5cx_api.h"
#include "vl53l5cx_plugin_motion_indicator.h"
#include "vl53l5cx_plugin_detection_thresholds.h"

#define UNUSED(x) (void)(x)

/* This function needs to be filled by the customer. It allows knowing when
 * the VL53L5CX interrupt is raised on GPIO1. This is the only way to use detection thresholds.
 */

static int WaitForL5Interrupt(VL53L5CX_Configuration * pDev) {

	//Add your implementation here ...
	UNUSED(pDev);

	return 0;
}

int example11(VL53L5CX_Configuration *p_dev)
{

	/*********************************/
	/*   VL53L5CX ranging variables  */
	/*********************************/

	uint8_t 		status, loop, isAlive, isReady, i;
	VL53L5CX_Motion_Configuration 	motion_config;	/* Motion configuration*/
	VL53L5CX_ResultsData 	Results;		/* Results data from VL53L5CX */


	/*********************************/
	/*   Power on sensor and init    */
	/*********************************/

	/* (Optional) Check if there is a VL53L5CX sensor connected */
	status = vl53l5cx_is_alive(p_dev, &isAlive);
	if(!isAlive || status)
	{
		printf("VL53L5CX not detected at requested address\n");
		return status;
	}

	/* (Mandatory) Init VL53L5CX sensor */
	status = vl53l5cx_init(p_dev);
	if(status)
	{
		printf("VL53L5CX ULD Loading failed\n");
		return status;
	}

	printf("VL53L5CX ULD ready ! (Version : %s)\n",
			VL53L5CX_API_REVISION);


	/*********************************/
	/*   Program motion indicator    */
	/*********************************/

	/* Create motion indicator with resolution 8x8 */
	status = vl53l5cx_motion_indicator_init(p_dev, &motion_config, VL53L5CX_RESOLUTION_8X8);
	if(status)
	{
		printf("Motion indicator init failed with status : %u\n", status);
		return status;
	}

	/* (Optional) Change the min and max distance used to detect motions. The
	 * difference between min and max must never be >1500mm, and minimum never be <400mm,
	 * otherwise the function below returns error 127 */
	status = vl53l5cx_motion_indicator_set_distance_motion(p_dev, &motion_config, 1000, 2000);
	if(status)
	{
		printf("Motion indicator set distance motion failed with status : %u\n", status);
		return status;
	}

	/* If user want to change the resolution, he also needs to update the motion indicator resolution */
	//status = vl53l5cx_set_resolution(p_dev, VL53L5CX_RESOLUTION_4X4);
	//status = vl53l5cx_motion_indicator_set_resolution(p_dev, &motion_config, VL53L5CX_RESOLUTION_4X4);


	/* Set the device in AUTONOMOUS and set a small integration time to reduce power consumption */
	status = vl53l5cx_set_resolution(p_dev, VL53L5CX_RESOLUTION_8X8);
	status = vl53l5cx_set_ranging_mode(p_dev, VL53L5CX_RANGING_MODE_AUTONOMOUS);
	status = vl53l5cx_set_ranging_frequency_hz(p_dev, 2);
	status = vl53l5cx_set_integration_time_ms(p_dev, 10);


	/*********************************/
	/*  Program detection thresholds */
	/*********************************/

	/* In this example, we want 1 thresholds per zone for a 8x8 resolution */
	/* Create array of thresholds (size cannot be changed) */
	VL53L5CX_DetectionThresholds thresholds[VL53L5CX_NB_THRESHOLDS];

	/* Set all values to 0 */
	memset(&thresholds, 0, sizeof(thresholds));

	/* Add thresholds for all zones (64 zones in resolution 4x4, or 64 in 8x8) */
	for(i = 0; i < 64; i++){
		thresholds[i].zone_num = i;
		thresholds[i].measurement = VL53L5CX_MOTION_INDICATOR;
		thresholds[i].type = VL53L5CX_GREATER_THAN_MAX_CHECKER;
		thresholds[i].mathematic_operation = VL53L5CX_OPERATION_NONE;

		/* The value 44 is given as example. All motion above 44 will be considered as a movement */
		thresholds[i].param_low_thresh = 44;
		thresholds[i].param_high_thresh = 44;
	}

	/* The last thresholds must be clearly indicated. As we have 64
	 * checkers, the last one is the 63 */
	thresholds[63].zone_num = VL53L5CX_LAST_THRESHOLD | thresholds[63].zone_num;

	/* Send array of thresholds to the sensor */
	vl53l5cx_set_detection_thresholds(p_dev, thresholds);

	/* Enable detection thresholds */
	vl53l5cx_set_detection_thresholds_enable(p_dev, 1);
	

	/*********************************/
	/*         Ranging loop          */
	/*********************************/

	status = vl53l5cx_start_ranging(p_dev);

	loop = 0;
	while(loop < 10)
	{
		/* Function WaitForL5Interrupt() does not exists, and must be
		 * implemented by user. It allows catching the interrupt raised on
		 * pin A3 (INT), when the checkers detect the programmed
		 * conditions.
		 */

		isReady = WaitForL5Interrupt(p_dev);

		if(isReady)
		{
			vl53l5cx_get_ranging_data(p_dev, &Results);

			/* As the sensor is set in 4x4 mode by default, we have a total
			 * of 16 zones to print. For this example, only the data of first zone are
			 * print */
			printf("Print data no : %3u\n", p_dev->streamcount);
			for(i = 0; i < 16; i++)
			{
				printf("Zone : %3d, Motion power : %6d\n",
					i,
					(int)Results.motion_indicator.motion[motion_config.map_id[i]]);
			}
			printf("\n");
			loop++;
		}

		/* Wait a few ms to avoid too high polling (function in platform
		 * file, not in API) */
		WaitMs(&p_dev->platform, 5);
	}

	status = vl53l5cx_stop_ranging(p_dev);
	printf("End of ULD demo\n");
	return status;
}
