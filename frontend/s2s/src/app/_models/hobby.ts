// Purpose: Interface for hobby and hobby type objects.
export interface Hobby {
    id: number;
    name: string;
    scenario_format: string;
    type: string;

}

export interface HobbyType {
    id: number;
    type: string;
}

export interface HobbyResponse {
    count: number;
    next: string;
    previous: string;
    results: Hobby[];
}

export interface HobbyTypeResponse {
    count: number;
    next: string;
    previous: string;
    results: HobbyType[];
}