/*******************************************************************************
Copyright (C) 2015, STMicroelectronics International N.V.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of STMicroelectronics nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS ARE DISCLAIMED.
IN NO EVENT SHALL STMICROELECTRONICS INTERNATIONAL N.V. BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************************/

#include <unistd.h>
#include <signal.h>
#include <dlfcn.h>

#include <stdio.h>
#include <string.h>

#include "vl53l5cx_api.h"

#include "examples.h"

VL53L5CX_Configuration sensors[5];
VL53L5CX_ResultsData Results;

int exit_main_loop = 0;

void sighandler(int signal)
{
	printf("SIGNAL Handler called, signal = %d\n", signal);
	exit_main_loop  = 1;
}

int init_sensor(int sensor)
{
	// VL53L5CX_Configuration Dev = sensors[sensor];

	int status = vl53l5cx_comms_init(&sensors[0].platform);
	if(status)
	{
		printf("VL53L5CX comms init failed\n");
		return -1;
	}
	status = vl53l5cx_init(&sensors[0]);

	//printf("%d \n",Dev.platform.address);
	if(status)
	{
		printf("VL53L5CX init failed\n");
		return -1;
	}

	return status;

}

int set_resolution(int sensor, int resolution){

	if(resolution == VL53L5CX_RESOLUTION_4X4)
	{
		printf("Resoltion 4x4 \n");
		return vl53l5cx_set_resolution(&sensors[0], VL53L5CX_RESOLUTION_4X4);
	} 
	else if(resolution == VL53L5CX_RESOLUTION_8X8)
	{
		printf("Resolution 8x8 \n");
		return vl53l5cx_set_resolution(&sensors[0], VL53L5CX_RESOLUTION_8X8);
	}
	else{
		printf("Error");
		return -1;
	}
	return 0;
}

int set_ranging_frequency(int sensor, uint8_t frequency){

	return vl53l5cx_set_ranging_frequency_hz(&sensors[sensor], frequency);

}

int set_target_order(int sensor, uint8_t target_order){

	return vl53l5cx_set_target_order(&sensors[sensor], target_order);	
}

int get_integration_time_ms(int sensor)
{
	int integration_time_ms;
	vl53l5cx_get_integration_time_ms(&sensors[sensor],&integration_time_ms);
	
	int ranging_mode;
	vl53l5cx_get_resolution(&sensors[sensor],&ranging_mode);
	printf("Ranging Mode %d \n",ranging_mode);


	return integration_time_ms;
}

int start_ranging(int sensor)
{
	return vl53l5cx_start_ranging(&sensors[sensor]);
}

int check_data_ready(int sensor)
{
	int isReady =0;

	while(isReady != 1)
	{
		vl53l5cx_check_data_ready(&sensors[sensor], &isReady);
		WaitMs(&sensors[sensor],5);
	}

	vl53l5cx_get_ranging_data(&sensors[sensor], &Results);
	
	return isReady;
}

int16_t get_ranging_data(int sensor, int target)
{
	// printf(" %d \n",Results.distance_mm[VL53L5CX_NB_TARGET_PER_ZONE*target]);
	return Results.distance_mm[VL53L5CX_NB_TARGET_PER_ZONE*target];	
}

int get_ranging_status(int sensor, int target)
{
	return Results.target_status[VL53L5CX_NB_TARGET_PER_ZONE*target];	
}

void stop_ranging(int sensor)
{
	vl53l5cx_stop_ranging(&sensors[sensor]);
}

int main(int argc, char ** argv)
{
	char choice[20];
	int status;
	VL53L5CX_Configuration 	Dev;

	/*********************************/
	/*   Power on sensor and init    */
	/*********************************/

	/* Initialize channel com */
	status = vl53l5cx_comms_init(&Dev.platform);
	if(status)
	{
		printf("VL53L5CX comms init failed\n");
		return -1;
	}

	printf("Starting examples with ULD version %s\n", VL53L5CX_API_REVISION);

	do {
		printf("----------------------------------------------------------------------------------------------------------\n");
		printf(" VL53L5CX uld driver test example menu \n");
		printf(" ------------------ Ranging menu ------------------\n");
		printf(" 1 : basic ranging\n");
		printf(" 2 : get and set params\n");
		printf(" 3 : change ranging mode\n");
		printf(" 4 : power down and up the device\n");
		printf(" 5 : multiple target per zone\n");
		printf(" 6 : I2C and  RAM optimization\n");
		printf(" 7 : (plugin) calibrate xtalk\n");
		printf(" 8 : (plugin) visualize xtalk calibration data\n");
		printf(" 9 : (plugin) detection thresholds - need to catch GPIO1 interrupt for this example\n");
		printf(" 10 : (plugin) motion indicator\n");
		printf(" 11 : (plugin) motion indicator with detection thresholds - need to catch GPIO1 interrupt for this example\n");
		printf(" 12 : exit\n");
		printf("----------------------------------------------------------------------------------------------------------\n");

		printf("Your choice ?\n ");
		scanf("%s", choice);

		if (strcmp(choice, "1") == 0) {
			printf("Starting Test 1\n");
			status = example1(&Dev);
			printf("\n");
		}
		else if (strcmp(choice, "2") == 0) {
			printf("Starting Test 2\n");
			status = example2(&Dev);
			printf("\n");
		}
		else if (strcmp(choice, "3") == 0) {
			printf("Starting Test 3\n");
			status = example3(&Dev);
			printf("\n");
		}
		else if (strcmp(choice, "4") == 0) {
			printf("Starting Test 4\n");
			status = example4(&Dev);
			printf("\n");
		}
		else if (strcmp(choice, "5") == 0) {
			printf("Starting Test 5\n");
			status = example5(&Dev);
			printf("\n");
		}
		else if (strcmp(choice, "6") == 0) {
			printf("Starting Test 6\n");
			status = example6(&Dev);
			printf("\n");
		}
		else if (strcmp(choice, "7") == 0) {
			printf("Starting Test 7\n");
			status = example7(&Dev);
			printf("\n");
		}
		else if (strcmp(choice, "8") == 0) {
			printf("Starting Test 8\n");
			status = example8(&Dev);
			printf("\n");
		}
		
		else if (strcmp(choice, "9") == 0) {
			printf("Starting Test 9\n");
			status = example9(&Dev);
			printf("\n");
		}
		
		else if (strcmp(choice, "10") == 0) {
			printf("Starting Test 10\n");
			status = example10(&Dev);
			printf("\n");
		}
		
		else if (strcmp(choice, "11") == 0) {
			printf("Starting Test 11\n");
			status = example11(&Dev);
			printf("\n");
		}
		
		else if (strcmp(choice, "12") == 0){
			exit_main_loop = 1;
		}
		
		else{
			printf("Invalid choice\n");
		}

	} while (!exit_main_loop);

	vl53l5cx_comms_close(&Dev.platform);

	return 0;
}
