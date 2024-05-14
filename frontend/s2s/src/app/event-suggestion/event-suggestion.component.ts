import { Component, OnInit, ChangeDetectorRef } from '@angular/core';

// services
import { SuggestionService } from '../_services/suggestion.service';
import { AuthService } from '../_services/auth.service';
import { EventService } from '../_services/event.service';

// helpers
import { Suggestion } from '../_models/suggestions';

/**
 * Component for displaying event suggestions.
 * 
 * This component displays event suggestions to users and allows them to RSVP to 
 * events.
 * 
 * @example
 * ```
 * <app-event-suggestion></app-event-suggestion>
 * ```
 * 
 * @see SuggestionService
 * @see AuthService
 * @see EventService
 */
@Component({
  selector: 'app-event-suggestion',
  templateUrl: './event-suggestion.component.html',
  styleUrl: './event-suggestion.component.css'
})
export class EventSuggestionComponent implements OnInit {
  showEventSuggestionDialog = false;
  currentSuggestion!: Suggestion;
  eventSuggestions: Suggestion[] = [];
  
  constructor (
    private suggestionService: SuggestionService,
    private authService: AuthService,
    private eventService: EventService
  ) {
    this.setDefaultEventSuggestion();
   }

  ngOnInit(): void {
    this.suggestionService.getSuggestions().subscribe(suggestions => {
      if (suggestions.length > 0) {
        this.eventSuggestions = suggestions;
        this.currentSuggestion = suggestions[0];
        this.showEventSuggestionDialog = true;
      }
    });
  }

  /**
   * Sets a default event suggestions.
   */
  setDefaultEventSuggestion() {
    this.currentSuggestion = {
      id: 0,
      user_id: 0,
      event_id: 0,
      probability_of_attendance: 0,
      event_name: '',
      event_description: '',
      event_date: '',
      event_duration: 0,
      event_max_attendees: 0,
      address1: '',
      address2: '',
      city: '',
      state: '',
      zipcode: '',
    };
  }

  /**
   * Closes the event suggestion dialog.
   */
  closeEventSuggestionDialog() {
    this.showEventSuggestionDialog = false;
    document.body.style.overflow = 'auto';
  }

  /**
   * Handles the user's RSVP submission.
   * 
   * @param rsvp - The user's RSVP.
   */
  onSubmitRSVP(rsvp: string) {
    // send RSVP to the API
    this.suggestionService.sendRSVP({
      user_id: this.authService.userValue.id,
      event_id: this.currentSuggestion.event_id,
      rsvp: rsvp,
    }).subscribe(() => {
      // remove the current suggestion from the list
      this.eventSuggestions.shift();
      this.eventService.loadAllEvents();

      // show the next suggestion, if applicable
      if (this.eventSuggestions.length > 0) {
        this.currentSuggestion = this.eventSuggestions[0];
      } else {
        this.showEventSuggestionDialog = false;
        document.body.style.overflow = 'auto';
      }
    });
  }
}
