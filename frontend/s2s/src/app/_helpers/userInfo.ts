export interface UserLocation {
    city: string,
    state: string, 
    zipCode: string,
    address1: string
}

export interface User {
    firstName: string,
    lastName: string,
    email: string,
    phoneNumber: number,
    username: string,
    id: number,
    onboarded: boolean
}