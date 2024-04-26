export interface Onboarding {
    user_id: number,
    onboarded: boolean,
    most_interested_hobbies: string[] | string,
    least_interested_hobbies: string[] | string,
    num_participants: string[] | string,
    distance: string,
    similarity_to_group: string,
    similarity_metrics: string[] | string,
    gender: string[] | string,
    gender_description: string,
    race: string[] | string,
    race_description: string,
    age: string,
    sexual_orientation: string,
    sexual_orientation_description: string,
    religion: string,
    religion_description: string,
    political_leaning: string,
    political_description: string,
    zip_code: string,
    city: string,
    state: string,
    address_line1: string,
    event_frequency: string,
    event_notifications: string,
}