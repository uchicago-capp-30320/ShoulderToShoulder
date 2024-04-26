// Description: Contains the scenario class.

import { Hobby } from "../_models/hobby"
import { SafeHtml } from "@angular/platform-browser";
import { ScenarioObj } from "../_models/scenarios";

/**
 * Interface for the scenarios object.
 */
export interface ScenarioInterface {
    id: number;
    description: SafeHtml;
}

/**
 * Class for generating a scenario object.
 * 
 * @param hobby The hobby to do.
 * @param duration The duration of the hobby.
 * @param day The day to do the hobby.
 * @param numPeople The number of people to do the hobby with.
 * @param mileage The mileage to travel to do the hobby.
 * 
 * @returns A scenario object describing the activity.
 */
export class Scenario {
    public scenarioObj: ScenarioObj;
    constructor(
        public hobby1: Hobby,
        public hobby2: Hobby,
        public time: string,
        public day: string,
        public numPeople: string,
        public mileage: string,
        private alteredVariable: string,
        public alteredVariableValue?: string
    ) {
        this.scenarioObj = {
            user_id: 0,
            hobby1: hobby1.name,
            hobby2: hobby2.name,
            distance1: mileage,
            distance2: mileage,
            num_participants1: numPeople,
            num_participants2: numPeople,
            day_of_week1: day,
            day_of_week2: day,
            time_of_day1: time,
            time_of_day2: time,
            prefers_event1: false,
            prefers_event2: false,
            };
    }

    /**
     * Generates a scenario template based on the altered variable.
     * 
     * @param altered The variable to vary in the scenario
     * @returns A string version of the scenario with the altered variable.
     */
    public getScenarioTemplate(altered: any) {
        // altering the time
        if (this.alteredVariable === "time") {
            this.scenarioObj.time_of_day2 = altered;
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1.scenario_format}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a <b>${this.day} ${this.time}</b>.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2.scenario_format}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a <b>${this.day} ${altered}</b>.<br><br> 
               `
        }
        
        // altering the day
        else if (this.alteredVariable === "day") {
            this.scenarioObj.day_of_week2 = altered;
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1.scenario_format}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a <b>${this.day} ${this.time}</b>.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2.scenario_format}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a <b>${altered} ${this.time}</b>.<br><br> 
               `
        }
            

        // altering the number of people
        else if (this.alteredVariable === "numPeople") {
            this.scenarioObj.num_participants2 = altered;
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1.scenario_format}</b> with <b>${this.numPeople}</b>
                other people at a location that is ${this.mileage} of you on a ${this.day} ${this.time}.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2.scenario_format}</b> with <b>${altered}</b>
                other people at a location that is ${this.mileage} of you on a ${this.day} ${this.time}.<br><br> 
               `
        }
            

        // altering the mileage
        else if (this.alteredVariable === "mileage") {
            this.scenarioObj.distance2 = altered;
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1.scenario_format}</b> with ${this.numPeople}
                other people at a location that is <b>${this.mileage}</b> of you on a ${this.day} ${this.time}.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2.scenario_format}</b> with ${this.numPeople}
                other people at a location that is <b>${altered}</b> of you on a ${this.day} ${this.time}.<br><br> 
               `
        }
            

        // error
        else
            return `Error: Altered variable ${altered} not found. Please enter
                    one of the following: hobby, time, day, numPeople, mileage.`;
    }
}