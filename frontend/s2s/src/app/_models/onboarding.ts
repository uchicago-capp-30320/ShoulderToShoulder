// Purpose: Interface for onboarding data.
export interface Onboarding {
    user_id: number | 0,
    onboarded: boolean | false,
    most_interested_hobby_types: number[] | [],
    most_interested_hobbies: number[] | [],
    least_interested_hobbies: number[] | [],
    num_participants: string[] | string | [],
    distance: string | '',
    similarity_to_group: string | '',
    similarity_metrics: string[] | string | [],
    pronouns: string | '',
    gender: string[] | string | [],
    gender_description: string | '',
    race: string[] | string | [],
    race_description: string | '',
    age: string | '',
    sexual_orientation: string | '',
    sexual_orientation_description: string | '',
    religion: string | '',
    religion_description: string | '',
    political_leaning: string | '',
    political_description: string | '',
    zip_code: string | '',
    city: string | '',
    state: string | '',
    address_line1: string | '',
    event_frequency: string | '',
    event_notification: string | '',
}

export interface OnboardingResp {
    count: number | 0,
    next: string | '',
    previous: string | '',
    results: Onboarding[]
}
