/**
 * An interface for representing a hobby that a user can select.
 */
export interface Hobby {
    id: number;
    name: string;
    scenario_format: string;
    max_participants: number;
    type: string;

}

export interface HobbyResponse {
    count: number;
    next: string;
    previous: string;
    results: Hobby[];
}