import { Component, OnInit } from '@angular/core';
import { formatDate } from '@angular/common';

// services
import { AuthService } from '../_services/auth.service';
import { OnboardingService } from '../_services/onboarding.service';
import { EventService } from '../_services/event.service';
import { GroupsService } from '../_services/groups.service';
import { HobbyService } from '../_services/hobbies.service';

// helpers
import { Event } from '../_models/event';
import { Hobby } from '../_models/hobby';

@Component({
  selector: 'app-profile-overview',
  templateUrl: './profile-overview.component.html',
  styleUrl: './profile-overview.component.css'
})
export class ProfileOverviewComponent implements OnInit {
  showAddlEventInformation: boolean = false;
  currentEvent?: Event;
  pastEvents: Event[] = []
  upcomingEvents: Event[] = []
  numEventsAttended: number = 0;

  constructor(
    public authService: AuthService,
    public onboardingService: OnboardingService,
    public eventService: EventService,
    public groupsService: GroupsService,
    public hobbyService: HobbyService
  ) { }

  ngOnInit(): void {
    this.eventService.loadAllEvents();
    this.eventService.pastEvents.subscribe(events => this.pastEvents = events);
    this.eventService.upcomingEvents.subscribe(events => this.upcomingEvents = events);
    this.eventService.numEventsAttended.subscribe(num => this.numEventsAttended = num);
  }

  get enjoys(): string {
    const max = 3;
    const hobbyIds = this.onboardingService.onboarding.most_interested_hobbies;
    let mostInterestedHobbies: Hobby[] = [];
    hobbyIds.map(id => this.hobbyService.hobbies.subscribe(
      hobbies => {
        let hobby = hobbies.find(hobby => hobby.id == id);
        if (hobby) {
          mostInterestedHobbies.push(hobby);
        
        }
      }));

    if (mostInterestedHobbies.length == 0) {
      return 'nothing';
    } else if (mostInterestedHobbies.length == 1) {
      return mostInterestedHobbies[0].name;
    } else if (mostInterestedHobbies.length == 2) {
      return `${mostInterestedHobbies[0].name} and ${mostInterestedHobbies[1].name}`;
    } else {
      let message = '';
      mostInterestedHobbies.forEach((hobby, i) => {
        if (i < max) {
          message += hobby.name + ', ';
        }});
        return `${message} and more`;
    }
  }

  arrayOfRating(rating: number): number[] {
    return Array(rating).fill(0).map((x, i) => i);
  }

  showAddlEventInformationDialog(event: Event): void {
    console.log(event);
    this.showAddlEventInformation = true;
    this.currentEvent = event;
  }

  formatDate(date: string): string {
    return formatDate(date, 'medium', 'en-US');
  }
}