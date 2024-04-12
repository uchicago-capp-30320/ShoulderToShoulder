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
    private _hobby: Hobby;
    private _time: string;
    private _day: string;
    private _numPeople: number;
    private _mileage: number;
    private _altered_variable: string;

    constructor(hobby: Hobby, duration: string, day: string, numPeople: number, mileage: number, altered_variable: string) {
        this._hobby = hobby;
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
        let people: string = this.numPeople === 1 ? "person" : "people";
        let mile: string = this.mileage === 1 ? "mile" : "miles";

        // altering the hobby
        if (this._altered_variable === "hobby") {
            altered = (altered as Hobby).scenarioForm;
            return `You receive an invitation for two different events.<br><br> 
                For <b>event 1</b>, you are invited to <b>${this.hobby}</b> with ${this.numPeople}
                other ${people} ${this.mileage} ${mile} away from you at ${this.time}
                on a ${this.day}.
                For <b>event 2</b>, you are invited to <b>${altered}</b> with ${this.numPeople}
                other ${people} ${this.mileage} ${mile} away from you at ${this.time} on a ${this.day}.
                <br><br>Which event would you rather attend?`;
        }
            
        // altering the time
        else if (this._altered_variable === "time")
            return `You receive an invitation for two different events.<br><br>
                For <b>event 1</b>, you are invited to ${this.hobby} with ${this.numPeople}
                other ${people} ${this.mileage} ${mile} away from you at <b>${this.time}</b>
                on a ${this.day}.
                For <b>event 2</b>, you are invited to ${this.hobby} with ${this.numPeople}
                other ${people} ${this.mileage} ${mile} away from you at <b>${altered}</b>
                on a ${this.day}.
                <br><br>Which event would you rather attend?`;
            
        // altering the day
        else if (this._altered_variable === "day")
            return `You receive an invitation for two different events.<br><br>
                For <b>event 1</b>, you are invited to ${this.hobby} with ${this.numPeople}
                other ${people} ${this.mileage} ${mile} away from you at ${this.time}
                on a <b>${this.day}</b>.
                For <b>event 2</b>, you are invited to ${this.hobby} with ${this.numPeople}
                other ${people} ${this.mileage} ${mile} away from you at ${this.time}
                on a <b>${altered}</b>.
                <br><br>Which event would you rather attend?`;

        // altering the number of people
        else if (this._altered_variable === "numPeople")
            return `You receive an invitation for two different events.<br><br>
                For <b>event 1</b>, you are invited to ${this.hobby} with <b>${this.numPeople}
                other ${people}</b> ${this.mileage} ${mile} away from you at ${this.time}
                on a ${this.day}.
                For <b>event 2</b>, you are invited to ${this.hobby} with <b>${altered}
                other ${people}</b> ${this.mileage} ${mile} away from you at ${this.time}
                on a ${this.day}.
                <br><br>Which event would you rather attend?`;

        // altering the mileage
        else if (this._altered_variable === "mileage")
            return `You receive an invitation for two different events.<br><br>
                For <b>event 1</b>, you are invited to ${this.hobby} with ${this.numPeople}
                other ${people} <b>${this.mileage} ${mile} away</b> from you at ${this.time}
                on a ${this.day}.
                For <b>event 2</b>, you are invited to ${this.hobby} with ${this.numPeople}
                other ${people} <b>${altered} ${mile} away</b> from you at ${this.time}
                on a ${this.day}.
                <br><br>Which event would you rather attend?`;

        // error
        else
            return `Error: Altered variable ${altered} not found. Please enter
                    one of the following: hobby, time, day, numPeople, mileage.`;
    }

    set hobby(hobby: Hobby) {
        this._hobby = hobby;
    }

    get hobby(): string {
        return this._hobby.scenarioForm;
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

    set numPeople(numPeople: number) {
        this._numPeople = numPeople;
    }

    get numPeople(): number {
        return this._numPeople;
    }

    set mileage(mileage: number) {
        this._mileage = mileage;
    }

    get mileage(): number {
        return this._mileage;
    }

    set altered_variable(altered_variable: string) {
        this._altered_variable = altered_variable;
    }

    get altered_variable(): string {
        return this._altered_variable;
    }
}