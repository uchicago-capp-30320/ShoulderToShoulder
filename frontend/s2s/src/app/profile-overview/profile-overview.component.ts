import { Component, OnInit } from '@angular/core';
import { formatDate } from '@angular/common';

// services
import { AuthService } from '../_services/auth.service';
import { OnboardingService } from '../_services/onboarding.service';
import { EventService } from '../_services/event.service';
import { HobbyService } from '../_services/hobbies.service';

// helpers
import { Event } from '../_models/event';
import { Hobby } from '../_models/hobby';

/**
 * Component to display user profile information.
 * 
 * This component displays the user's profile information, including their name, 
 * email, and profile picture.
 * 
 * @example
 * ```
 * <app-profile></app-profile>
 * ```
 * 
 * @remarks
 * This component relies on the AuthService, OnboardingService, EventService, and 
 * HobbyService to manage user profile data.
 * 
 * @see AuthService
 * @see OnboardingService
 * @see EventService
 * @see HobbyService
 * 
 * @export
 * @class ProfileOverviewComponent
 */
@Component({
  selector: 'app-profile-overview',
  templateUrl: './profile-overview.component.html',
  styleUrl: './profile-overview.component.css'
})
export class ProfileOverviewComponent implements OnInit {
  showAddlEventInformation: boolean = false;
  attended: boolean = false;
  rating: number = 0;
  pastEvent: boolean = false;
  currentEvent?: Event;
  pastEvents: Event[] = []
  upcomingEvents: Event[] = []
  numEventsAttended: number = 0;

  constructor(
    public authService: AuthService,
    public onboardingService: OnboardingService,
    public eventService: EventService,
    public hobbyService: HobbyService
  ) { }

  ngOnInit(): void {
    this.eventService.loadAllEvents();
    this.eventService.pastEvents.subscribe(events => this.pastEvents = events);
    this.eventService.upcomingEvents.subscribe(events => this.upcomingEvents = events);
    this.eventService.numEventsAttended.subscribe(num => this.numEventsAttended = num);
  }

  /**
   * Gets the user's most enjoyed hobbies.
   */
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

  /**
   * Gets the user's least enjoyed hobbies.
   */
  arrayOfRating(rating: number): number[] {
    return Array(rating).fill(0).map((x, i) => i);
  }

  /**
   * Gets the user's least enjoyed hobbies.
   */
  showAddlEventInformationDialog(event: Event): void {
    this.showAddlEventInformation = true;
    this.currentEvent = event;
  }

  showPastEventInformationDialog(event: Event): void {
    this.showAddlEventInformation = true;
    this.currentEvent = event;
    this.pastEvent = true;
  }

  closeEventDialog(): void {
    this.showAddlEventInformation = false;
    this.pastEvent = false;  // Resetting this if it's used to determine dialog behavior
  }

  submitReview(): void {
    if (this.currentEvent) {
      this.eventService.reviewEvent(this.currentEvent, this.rating, this.attended).subscribe(
        event => {
          this.currentEvent = event;
          this.eventService.loadAllEvents();
        });
    }
    this.pastEvent = false;
    this.showAddlEventInformation = false;
    this.rating = 0;
    this.attended = false;
  }


  /**
   * Closes the additional event information dialog.
   */
  formatDate(date: string): string {
    return formatDate(date, 'medium', 'en-US');
  }
}