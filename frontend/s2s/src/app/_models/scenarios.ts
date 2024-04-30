export var maxScenarios = 10;

export interface ScenarioObj {
    user_id: number,
    hobby1: number,
    hobby2: number,
    distance1: string,
    distance2: string,
    num_participants1: string,
    num_participants2: string,
    day_of_week1: string,
    day_of_week2: string,
    time_of_day1: string,
    time_of_day2: string,
    prefers_event1: boolean,
    prefers_event2: boolean,
    duration_h1: number,
    duration_h2: number
}