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
    userStorage = localStorage.getItem('user');
    user = this.userStorage ? JSON.parse(this.userStorage) : {id: 0};
    timeOfDayMap: { [index: string]: string } = {
        "morning": "Morning (9a-12p)",
        "afternoon": "Afternoon (1-4p)",
        "evening": "Evening (5-8p)",
        "night": "Night (9p-12a)"
    }
    distanceMap: { [index: string]: string } = {
        "within 1 mile": "Within 1 mile",
        "within 5 miles": "Within 5 miles",
        "within 10 miles": "Within 10 miles",
        "within 15 miles": "Within 15 miles", 
        "within 20 miles": "Within 20 miles",
        "within 30 miles": "Within 30 miles",
        "within 40 miles": "Within 40 miles",
        "within 50 miles": "Within 50 miles",
    }

    public scenarioObj: ScenarioObj = {
        user_id: this.user.id,
        hobby1: this.hobby1.id,
        hobby2: this.hobby2.id,
        distance1: this.distanceMap[this.mileage.toLowerCase()],
        distance2: this.distanceMap[this.mileage.toLowerCase()],
        num_participants1: this.numPeople,
        num_participants2: this.numPeople,
        day_of_week1: this.day,
        day_of_week2: this.day,
        time_of_day1: this.timeOfDayMap[this.time],
        time_of_day2: this.timeOfDayMap[this.time],
        prefers_event1: false,
        prefers_event2: false,
        duration_h1: this.duration,
        duration_h2: this.duration
    };

    constructor(
        public hobby1: Hobby,
        public hobby2: Hobby,
        public time: string,
        public day: string,
        public numPeople: string,
        public mileage: string,
        public duration: number,
        private alteredVariable: string,
        public alteredVariableValue?: string
    ) {}

    /**
     * Generates a scenario template based on the altered variable.
     * 
     * @param altered The variable to vary in the scenario
     * @returns A string version of the scenario with the altered variable.
     */
    public getScenarioTemplate(altered: any) {
        // altering the time
        if (this.alteredVariable === "time") {
            this.scenarioObj.time_of_day2 = this.timeOfDayMap[altered];
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1.scenario_format}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a <b>${this.day} ${this.time}</b> 
                for up to ${this.duration} hours.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2.scenario_format}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a <b>${this.day} ${altered}</b>
                for up to ${this.duration} hours.<br><br> 
               `
        }
        
        // altering the day
        else if (this.alteredVariable === "day") {
            this.scenarioObj.day_of_week2 = altered;
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1.scenario_format}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a <b>${this.day} ${this.time}</b>
                for up to ${this.duration} hours.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2.scenario_format}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a <b>${altered} ${this.time}</b>
                for up to ${this.duration} hours.<br><br> 
               `
        }   

        // altering the number of people
        else if (this.alteredVariable === "numPeople") {
            this.scenarioObj.num_participants2 = altered;
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1.scenario_format}</b> with <b>${this.numPeople}</b>
                other people at a location that is ${this.mileage} of you on a ${this.day} ${this.time}
                for up to ${this.duration} hours.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2.scenario_format}</b> with <b>${altered}</b>
                other people at a location that is ${this.mileage} of you on a ${this.day} ${this.time}
                for up to ${this.duration} hours.<br><br> 
               `
        }
            
        // altering the mileage
        else if (this.alteredVariable === "mileage") {
            this.scenarioObj.distance2 = this.distanceMap[altered.toLowerCase()];
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1.scenario_format}</b> with ${this.numPeople}
                other people at a location that is <b>${this.mileage}</b> of you on a ${this.day} ${this.time}
                for up to ${this.duration} hours.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2.scenario_format}</b> with ${this.numPeople}
                other people at a location that is <b>${altered}</b> of you on a ${this.day} ${this.time}
                for up to ${this.duration} hours.<br><br> 
               `
        }

        // altering the duration
        else if (this.alteredVariable === "duration") {
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1.scenario_format}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a ${this.day} ${this.time}
                for up to <b>${this.duration} hours</b>.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2.scenario_format}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a ${this.day} ${this.time}
                for up to <b>${altered} hours</b>.<br><br> 
                `
        }
            
        // error
        else
            return `Error: Altered variable ${altered} not found. Please enter
                    one of the following: hobby, time, day, numPeople, mileage.`;
    }
}