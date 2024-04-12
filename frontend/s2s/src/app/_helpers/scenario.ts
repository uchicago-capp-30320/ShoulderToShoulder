// Purpose: Contains the scenario class.

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
    private _hobby: string;
    private _duration: string;
    private _day: string;
    private _numPeople: number;
    private _mileage: number;

    constructor(hobby: string, duration: string, day: string, numPeople: number, mileage: number) {
        this._hobby = hobby;
        this._duration = duration;
        this._day = day;
        this._numPeople = numPeople;
        this._mileage = mileage;
    }

    get scenarioTemplate(): string {
        return `I want to do this activity ${this._hobby} from ${this._duration} on ${this._day}.`;
    }

    set hobby(hobby: string) {
        this._hobby = hobby;
    }

    get hobby(): string {
        return this._hobby;
    }

    set duration(duration: string) {
        this._duration = duration;
    }

    get duration(): string {
        return this._duration;
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
}