// Purpose: Interface for User object.

export interface UserLocation {
    city: string,
    state: string, 
    zipCode: string,
    address1: string
}

export interface UserSignUp {
    first_name: string,
    last_name: string,
    email: string,
    password: string,

}

export interface User {
    first_name: string,
    last_name: string,
    email: string,
    username: string,
    id: number,
}

export interface UserResponse {
    user: User,
    access_token: string
    refresh_token: string
}