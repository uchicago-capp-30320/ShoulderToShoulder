// Purpose: Contains the scenario class.

import { Hobby } from "./preferences";
import { SafeHtml } from "@angular/platform-browser";

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
    private _hobby1: Hobby;
    private _hobby2: Hobby;
    private _time: string;
    private _day: string;
    private _numPeople: string;
    private _mileage: string;
    private _altered_variable: string;

    constructor(hobby1: Hobby, hobby2: Hobby, duration: string, day: string, numPeople: string, mileage: string, altered_variable: string) {
        this._hobby1 = hobby1;
        this._hobby2 = hobby2;
        this._time = duration;
        this._day = day;
        this._numPeople = numPeople;
        this._mileage = mileage;
        this._altered_variable = altered_variable;
    }

    /**
     * Generates a scenario template based on the altered variable.
     * 
     * @param altered The variable to vary in the scenario
     * @returns A string version of the scenario with the altered variable.
     */
    public getScenarioTemplate(altered: any) {
        // altering the time
        if (this._altered_variable === "time")
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a ${this.day} <b>${this.time}</b>.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a ${this.day} <b>${altered}</b>.<br><br> 
               <b>Which event would you rather attend?</b>`;
        
        // altering the day
        else if (this._altered_variable === "day")
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a <b>${this.day}</b> ${this.time}.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2}</b> with ${this.numPeople}
                other people at a location that is ${this.mileage} of you on a <b>${altered}</b> ${this.time}.<br><br> 
               <b>Which event would you rather attend?</b>`;

        // altering the number of people
        else if (this._altered_variable === "numPeople")
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1}</b> with <b>${this.numPeople}</b>
                other people at a location that is ${this.mileage} of you on a ${this.day} ${this.time}.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2}</b> with <b>${altered}</b>
                other people at a location that is ${this.mileage} of you on a ${this.day} ${this.time}.<br><br> 
               <b>Which event would you rather attend?</b>`;

        // altering the mileage
        else if (this._altered_variable === "mileage")
            return `You receive an invitation for two different events.<br><br>
                <b>Event 1</b>: You are invited to <b>${this.hobby1}</b> with ${this.numPeople}
                other people at a location that is <b>${this.mileage}</b> of you on a ${this.day} ${this.time}.<br><br> 
                <b>Event 2</b>: You are invited to <b>${this.hobby2}</b> with ${this.numPeople}
                other people at a location that is <b>${altered}</b> of you on a ${this.day} ${this.time}.<br><br> 
               <b>Which event would you rather attend?</b>`;

        // error
        else
            return `Error: Altered variable ${altered} not found. Please enter
                    one of the following: hobby, time, day, numPeople, mileage.`;
    }

    set hobby1(hobby: Hobby) {
        this._hobby1 = hobby;
    }

    get hobby1(): string {
        return this._hobby1.scenarioForm;
    }

    set hobby2(hobby: Hobby) {
        this._hobby2 = hobby;
    }

    get hobby2(): string {
        return this._hobby2.scenarioForm;
    }

    set duration(duration: string) {
        this._time = duration;
    }

    get time(): string {
        return this._time;
    }

    set day(day: string) {
        this._day = day;
    }

    get day(): string {
        return this._day;
    }

    set numPeople(numPeople: string) {
        this._numPeople = numPeople;
    }

    get numPeople(): string {
        return this._numPeople;
    }

    set mileage(mileage: string) {
        this._mileage = mileage;
    }

    get mileage(): string {
        return this._mileage;
    }

    set altered_variable(altered_variable: string) {
        this._altered_variable = altered_variable;
    }

    get altered_variable(): string {
        return this._altered_variable;
    }
}