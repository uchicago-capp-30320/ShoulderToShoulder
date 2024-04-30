import { Injectable } from '@angular/core';

export interface Group {
  name: string;
  numMembers: number;
  link: string;
}

@Injectable({
  providedIn: 'root'
})
export class GroupsService {
  // test data
  groups: Group[] = [
    {
      name: 'Group 1',
      numMembers: 3,
      link: 'https://example.com/group1'
    },
    {
      name: 'Group 2',
      numMembers: 5,
      link: 'https://example.com/group2'
    }
  ]

  constructor() { }
}
