/**
 * An interface for representing a hobby that a user can select.
 */
export interface Hobby {
    name: string;
    scenarioForm: string;
    maxParticipants: number;
    type: string;

}

export interface HobbyResponse {
    count: number;
    next: string;
    previous: string;
    results: Hobby[];
}