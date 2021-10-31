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
/*    VL53L5CX ULD ranging mode    */
/***********************************/
/*
* This example shows the differences between ranging modes of VL53L5CX
* (mode continuous and autonomous). For both modes, it initializes the VL53L5CX
* ULD, set the mode, and starts a ranging to capture 10 frames.
*
* In this example, we also suppose that the number of target per zone is
* set to 1 , and all output are enabled (see file platform.h).
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "vl53l5cx_api.h"

int example3(VL53L5CX_Configuration *p_dev)
{

	/*********************************/
	/*   VL53L5CX ranging variables  */
	/*********************************/

	uint8_t 				status, loop, isAlive, isReady, i;
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
	/*  Set ranging mode autonomous  */
	/*********************************/

	status = vl53l5cx_set_ranging_mode(p_dev, VL53L5CX_RANGING_MODE_AUTONOMOUS);
	if(status)
	{
		printf("vl53l5cx_set_ranging_mode failed, status %u\n", status);
		return status;
	}

	/* Using autonomous mode, the integration time can be updated (not possible
	 * using continuous) */
	status = vl53l5cx_set_integration_time_ms(p_dev, 20);

	/* Start a ranging session */
   	status = vl53l5cx_start_ranging(p_dev);
   	printf("Start ranging autonomous\n");

   	loop = 0;
   	while(loop < 10)
   	{
   		status = vl53l5cx_check_data_ready(p_dev, &isReady);
   		if(isReady)
   		{
   			vl53l5cx_get_ranging_data(p_dev, &Results);

   			/* As the sensor is set in 4x4 mode by default, we have a total
			 * of 16 zones to print. For this example, only the data of first zone are
			 * print */
   			printf("Print data no : %3u\n", p_dev->streamcount);
   			for(i = 0; i < 16; i++)
   			{
				printf("Zone : %3d, Status : %3u, Distance : %4d mm\n",
					i,
					Results.target_status[VL53L5CX_NB_TARGET_PER_ZONE*i],
					Results.distance_mm[VL53L5CX_NB_TARGET_PER_ZONE*i]);
   			}
   			printf("\n");
   			loop++;
   		}

		/* Wait a few ms to avoid too high polling (function in platform
		 * file, not in API) */
   		WaitMs(&p_dev->platform, 5);
   	}

   	status = vl53l5cx_stop_ranging(p_dev);
   	printf("Stop ranging autonomous\n");

	
	/*********************************/
	/* Set ranging mode continuous   */
	/*********************************/

	/* In continuous mode, the integration time cannot be programmed
	 * (automatically set to maximum value) */

	status = vl53l5cx_set_ranging_mode(p_dev, VL53L5CX_RANGING_MODE_CONTINUOUS);
	if(status)
	{
		printf("vl53l5cx_set_ranging_mode failed, status %u\n", status);
		return status;
	}

	/* Trying to update value below will have no impact on integration time */
	//status = vl53l5cx_set_integration_time_ms(p_dev, 20);

	/* Start a ranging session */
   	status = vl53l5cx_start_ranging(p_dev);
   	printf("Start ranging continuous\n");

   	loop = 0;
   	while(loop < 10)
   	{
   		status = vl53l5cx_check_data_ready(p_dev, &isReady);
   		if(isReady)
   		{
   			vl53l5cx_get_ranging_data(p_dev, &Results);

   			/* As the sensor is set in 4x4 mode by default, we have a total
			 * of 16 zones to print */
   			printf("Print data no : %3u\n", p_dev->streamcount);
   			for(i = 0; i < 16; i++)
   			{
   				printf("Zone : %3d, Status : %3u, Distance : %4d mm\n",
					i,
					Results.target_status[VL53L5CX_NB_TARGET_PER_ZONE*i],
					Results.distance_mm[VL53L5CX_NB_TARGET_PER_ZONE*i]);
   			}
   			printf("\n");
   			loop++;
   		}

		/* Wait a few ms to avoid too high polling (function in platform
		 * file, not in API) */
   		WaitMs(&p_dev->platform, 5);
   	}

   	status = vl53l5cx_stop_ranging(p_dev);
   	printf("Stop ranging continuous\n");

   	printf("End of ULD demo\n");
   	return status;
}
